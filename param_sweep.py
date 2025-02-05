import deephyper as dh
from deephyper.hpo import HpProblem
from deephyper.evaluator import profile
import torch
from utils.YParams import YParams
import logging
from train_mp_mod import train
import argparse
from typing import Union, Dict
import os
import pandas as pd
from deephyper.evaluator.callback import TqdmCallback
from deephyper.evaluator import Evaluator
from functools import partial

def create_problem(fixed_param=None, fixed_value=None, objective_type="rmse"):
    problem = HpProblem()
    
    # Dictionary of parameter ranges
    param_ranges = {
        "num_heads": ((4, 32), 8),  # (range, default)
        "depth": ((6, 24), 12),
        "embed_dim": ((128, 1024), 384),
        "num_years": ((1, 25), 15),
        "learning_rate": ((1e-5, 1e-3), 5e-4)  # Added learning rate
    }
    
    # Add hyperparameters, skipping the fixed one
    for param_name, (param_range, default) in param_ranges.items():
        if param_name == fixed_param:
            # Add as constant if it's the fixed parameter
            problem.add_hyperparameter([fixed_value], param_name, default_value=fixed_value)
        else:
            problem.add_hyperparameter(param_range, param_name, default_value=default)
    
    # Define objectives based on type
    if objective_type == "multi":
        problem.add_objective("rmse", minimize=True)
        problem.add_objective("memory_usage", minimize=True)
        problem.add_objective("iter_time", minimize=True)
    else:
        problem.add_objective("rmse", minimize=True)
    
    return problem

def run(config: dict, local_rank: int, world_rank: int, world_size: int) -> Union[float, Dict[str, float]]:
    params = YParams("./config/ViT.yaml", "mp")
    
    # Update model parameters
    params.num_heads = config["num_heads"]
    params.depth = config["depth"]
    params.embed_dim = config["embed_dim"]
    params.lr = config["learning_rate"]
    
    # Set shorter training for hyperparameter search
    params.num_iters = 1000
    params.num_epochs = params.num_iters // 32
    
    # Create args with proper parallelism settings
    args = argparse.Namespace()
    args.local_batch_size = 32
    args.enable_fused = True
    args.enable_jit = True
    args.noddp = False
    args.bucket_cap_mb = 25
    
    # Parallelism settings
    args.tensor_parallel = torch.cuda.device_count()
    args.context_parallel = 1
    args.parallel_order = "tp-cp-dp"
    
    try:
        val_rmse, memory_gb, iter_time = train(
            params, args,
            local_rank=local_rank,
            world_rank=world_rank,
            world_size=world_size,
            hyperparameter_search=True
        )
        
        if isinstance(HpProblem.objectives, dict):
            return {
                "rmse": float(val_rmse),
                "memory_usage": float(memory_gb),
                "iter_time": float(iter_time)
            }
        else:
            return float(val_rmse)
    except Exception as e:
        logging.error(f"Training failed with config {config}: {str(e)}")
        return float("inf")

class SaveResultsCallback(TqdmCallback):
    def __init__(self, results_path, world_rank=0, distributed=False):
        super().__init__()
        self.results_path = results_path
        self.world_rank = world_rank
        self.distributed = distributed
        
    def on_result(self, result):
        super().on_result(result)
        # Only rank 0 saves results
        if self.world_rank == 0:
            df = self.evaluator.search.results
            df.to_csv(self.results_path, index=False)
        
        # Synchronize processes after saving
        if self.distributed:
            torch.distributed.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-param', type=str, 
                       choices=['num_heads', 'depth', 'embed_dim', 'num_years', 'learning_rate'],
                       help='Parameter to hold constant')
    parser.add_argument('--fix-value', type=float, help='Value to hold the parameter at')  # Changed to float for learning rate
    parser.add_argument('--max-evals', type=int, default=100, help='Maximum number of evaluations')
    parser.add_argument('--objective', type=str, default="rmse", 
                       choices=["rmse", "multi"], help='Optimization objective type')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from previous results if they exist')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory to store results')
    args = parser.parse_args()
    
    # Get SLURM environment variables if available
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    world_rank = int(os.environ.get('SLURM_PROCID', 0))
    distributed = world_size > 1
    
    # Create results directory if it doesn't exist
    if world_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
    if distributed:
        torch.distributed.barrier()
    
    # Create filename for results
    filename = "hyperparameter_search_results"
    if args.fix_param:
        filename += f"_{args.fix_param}_{args.fix_value}"
    filename += ".csv"
    results_path = os.path.join(args.results_dir, filename)
    
    # Load existing results if resuming
    if args.resume and os.path.exists(results_path):
        if world_rank == 0:
            existing_results = pd.read_csv(results_path)
            start_eval = len(existing_results)
            logging.info(f"Resuming from evaluation {start_eval}")
        if distributed:
            torch.distributed.barrier()
    else:
        existing_results = None
        start_eval = 0
    
    problem = create_problem(args.fix_param, args.fix_value, args.objective)
    
    # Create callback that handles both progress bar and saving
    save_callback = SaveResultsCallback(
        results_path=results_path,
        world_rank=world_rank,
        distributed=distributed
    )
    
    # Create partial function with fixed rank arguments
    run_with_ranks = partial(run, 
                           local_rank=world_rank,
                           world_rank=world_rank,
                           world_size=world_size)
    
    evaluator = Evaluator.create(
        run_with_ranks,
        method="thread",
        method_kwargs={
            "num_workers": 1,
            "callbacks": [save_callback]
        }
    )
    
    search = dh.CBO(
        problem,
        evaluator,
        random_state=42,
        surrogate_model="ET",
        surrogate_model_kwargs={
            "n_estimators": 25,
            "min_samples_split": 8,
        },
        multi_point_strategy="qUCBd" if args.objective == "multi" else None
    )
    
    # Calculate remaining evaluations
    remaining_evals = args.max_evals - start_eval
    
    try:
        results = search.search(max_evals=remaining_evals)
        
        # Log best configuration found
        best_config = results.iloc[results['objective'].argmin()]
        logging.info(f"Best configuration found: {best_config.to_dict()}")
    except Exception as e:
        logging.error(f"Search interrupted: {str(e)}")
        logging.info("Results saved incrementally in: " + results_path)

    if distributed:
        torch.distributed.barrier()

if __name__ == "__main__":
    main()