import sys
import os
import time
import numpy as np
import argparse
import pynvml

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp

import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils import get_data_loader_distributed
from utils import comm
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse
from networks import vit

from distributed.mappings import init_ddp_model_and_reduction_hooks
from distributed.helpers import init_params_for_shared_weights

from utils.plots import generate_images
from pathlib import Path
import json
from datetime import datetime, timedelta
import subprocess

scratch = os.getenv("SCRATCH")
temp_train = Path(f"{scratch}/temp_train")
temp_val = Path(f"{scratch}/temp_val")

def data_subset(n_train: int=25):
    target = Path('/pscratch/sd/s/shas1693/data/sc24_tutorial_data')
    all_data = list((target/'train').iterdir())
    all_data = sorted(all_data)
    train_subset = all_data[:n_train]

    (temp_train/str(n_train)).mkdir(exist_ok=True, parents=True)
    (temp_val/str(n_train)).mkdir(exist_ok=True, parents=True)

    for x in train_subset:
        if not (temp_train/str(n_train)/x.name).exists():
            os.symlink(x, temp_train/str(n_train)/x.name)
    
    for x in (target/'valid').iterdir():
        if not (temp_val/str(n_train)/x.name).exists():
            os.symlink(x, temp_val/str(n_train)/x.name)


def clean_up_temp_dirs(n_train: int):
    for x in (temp_train/str(n_train)).iterdir():
        os.unlink(x)

    for x in (temp_val/str(n_train)).iterdir():
        os.unlink(x)
    (temp_val/str(n_train)).rmdir()
    (temp_train/str(n_train)).rmdir()
    

def get_remaining_time():
    """Get remaining time in seconds from SLURM environment variables"""
    if 'SLURM_JOB_ID' not in os.environ:
        logging.info("Not running in SLURM environment")
        return float('inf')
    
    try:
        # Get end and start times from SLURM
        end_time = int(os.environ.get('SLURM_JOB_END_TIME', 0))
        if end_time == 0:
            return float('inf')
            
        # Calculate remaining time
        remaining = max(0, end_time - time.time())
        
        return remaining
        
    except Exception as e:
        logging.warning(f"Failed to calculate remaining time: {e}")
        return float('inf')

def save_and_exit(model, optimizer, scheduler, iters, params, args, world_rank):
    """Save checkpoint and exit gracefully"""
    try:
        save_checkpoint(model, optimizer, scheduler, iters, params, args, world_rank)
        if world_rank == 0:
            logging.info("Time limit approaching - saved checkpoint and exiting")
        if params.distributed:
            torch.distributed.barrier()  # Ensure all processes finish saving
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error during save_and_exit: {e}")
        sys.exit(1)

def train(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank)

    # init pynvml and get handle
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    # get data loader
    logging.info("rank %d, begin data loader init" % world_rank)

    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(
        params, str(temp_train/str(params.n_train)), params.distributed, train=True
    )
    val_data_loader, valid_dataset = get_data_loader_distributed(
        params, str(temp_val/str(params.n_train)), params.distributed, train=False
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

    # Sanity check: Log GPU details
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).total / (1024 ** 3)
    logging.info(f"Rank {world_rank}: Using GPU {local_rank} - {gpu_name}, Total GPU memory: {total_memory:.2f} GB")

    # Log node details
    logging.info(f"Rank {world_rank}/{world_size} running on node: {os.uname().nodename}")

    # Distributed all-reduce test
    test_tensor = torch.tensor(world_rank, device=device)
    torch.distributed.all_reduce(test_tensor, op=torch.distributed.ReduceOp.SUM)
    if world_rank == 0:
        expected_sum = (world_size * (world_size - 1)) // 2
        logging.info(f"Sanity check: Sum of ranks across nodes: {test_tensor.item()} (Expected: {expected_sum})")

    # create model
    model = vit.ViT(params).to(device)

    if params.enable_jit:
        model = torch.compile(model)

    if params.amp_dtype == torch.float16:
        scaler = GradScaler()

    # weight initialization needs to be synced across shared weights
    if comm.get_size("tp-cp") > 1:
        init_params_for_shared_weights(model)

    if params.distributed and not args.noddp:
        model = init_ddp_model_and_reduction_hooks(model, device_ids=[local_rank],
                                                   output_device=[local_rank],
                                                   bucket_cap_mb=args.bucket_cap_mb)

    if params.enable_fused:
        optimizer = optim.Adam(
            model.parameters(), lr=params.lr, fused=True, betas=(0.9, 0.95)
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if world_rank == 0:
        logging.info(model)
        all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
            1024.0 * 1024.0 * 1024.0
        )
        logging.info(f"Scaffolding memory high watermark: {all_mem_gb} GB.")
        with open(f'{params.experiment_dir}/hparams.json', 'r') as file:
            data = json.load(file)
        data['parameters'] = param_count
        with open(f'{params.experiment_dir}/hparams.json', 'w') as file:
            json.dump(data, file, indent=4)

    # Calculate iterations for budget
    if params.budget:
        # Calculate sequence length
        seq_len = (360 // params.patch_size) * (720 // params.patch_size)
        logging.info(f'seq_len {seq_len}')
        # Number of iterations to run based on desired flops
        tokens_per_step = params.global_batch_size * seq_len
        logging.info(f'tokens_per_step {tokens_per_step}')
        max_steps = int(params.budget // (6 * param_count * tokens_per_step))
        logging.info(f'param_count: {param_count}')
        logging.info(f'max_steps {params.budget} / {(6 * param_count * tokens_per_step)}')
        params.num_iters = max_steps // tokens_per_step
        logging.info(f'Calculated {params.num_iters} iterations for compute budget {params.budget}')
        logging.info(f'train_data_loader: {len(train_data_loader)}')

    iters = 0
    startEpoch = 0

    if params.lr_schedule == "cosine":
        if params.warmup > 0:
            lr_scale = lambda x: min(
                (x + 1) / params.warmup,
                0.5 * (1 + np.cos(np.pi * x / params.num_iters)),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params.num_iters
            )
    else:
        scheduler = None

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss

    if world_rank == 0:
        logging.info("Starting Training Loop...")

    # Log initial loss on train and validation to tensorboard
    with torch.no_grad():
        inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
        gen = model(inp)
        tr_loss = loss_func(gen, tar)
        inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
        gen = model(inp)
        val_loss = loss_func(gen, tar)
        val_rmse = weighted_rmse(gen, tar)
        if params.distributed:
            torch.distributed.all_reduce(
                tr_loss, op=ReduceOp.AVG, group=comm.get_group("dp")
            )
            torch.distributed.all_reduce(
                val_loss, op=ReduceOp.AVG, group=comm.get_group("dp")
            )
            torch.distributed.all_reduce(
                val_rmse, op=ReduceOp.AVG, group=comm.get_group("dp")
            )
        if world_rank == 0:
            args.tboard_writer.add_scalar("Loss/train", tr_loss.item(), 0)
            args.tboard_writer.add_scalar("Loss/valid", val_loss.item(), 0)
            args.tboard_writer.add_scalar(
                "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], 0
            )
    
    params.num_epochs = params.num_iters // len(train_data_loader)

    iters = 0
    t1 = time.time()
    # Track start time and time limit
    start_time = time.time()
    time_buffer = args.time_buffer  # Use command line argument instead of hardcoded value
    
    # Set default logging frequency if not specified
    if not hasattr(params, 'logging_freq'):
        params.logging_freq = 100  # Log every 100 iterations by default
        if world_rank == 0:
            logging.info(f"Setting default logging frequency to {params.logging_freq}")
    
    # Training loop
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []
        tr_time = 0.0
        dat_time = 0.0
        log_time = 0.0

        model.train()
        step_count = 0

        for i, data in enumerate(train_data_loader, 0):
            if iters >= params.num_iters:
                break  
            if world_rank == 0:
                if epoch == 3 and i == 0:
                    torch.cuda.profiler.start()
                if epoch == 3 and i == len(train_data_loader) - 1:
                    torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            dat_start = time.time()
            torch.cuda.nvtx.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            torch.cuda.nvtx.range_pop()  # copy in

            tr_start = time.time()
            b_size = inp.size(0)

            optimizer.zero_grad()

            torch.cuda.nvtx.range_push(f"forward")
            with autocast(enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)
            torch.cuda.nvtx.range_pop()  # forward

            if world_rank == 0 and i == 1:  # print the mem used
                all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
                    1024.0 * 1024.0 * 1024.0
                )
                logging.info(f" Memory usage after forward pass: {all_mem_gb} GB.")

            if params.amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                torch.cuda.nvtx.range_push(f"optimizer")
                scaler.step(optimizer)
                torch.cuda.nvtx.range_pop()  # optimizer
                scaler.update()
            else:
                loss.backward()
                torch.cuda.nvtx.range_push(f"optimizer")
                optimizer.step()
                torch.cuda.nvtx.range_pop()  # optimizer

            if params.distributed:
                torch.distributed.all_reduce(
                    loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                )
            tr_loss.append(loss.item())

            torch.cuda.nvtx.range_pop()  # step
            # lr step
            scheduler.step()

            tr_end = time.time()
            tr_time += tr_end - tr_start
            dat_time += tr_start - dat_start
            step_count += 1
            iters += 1

            # Check remaining time
            if world_rank == 0:
                remaining_time = torch.tensor(get_remaining_time(), device=device)
            else:
                remaining_time = torch.tensor(0.0, device=device)
                
            if params.distributed:
                torch.distributed.broadcast(remaining_time, src=0)
            
            if remaining_time.item() < time_buffer:
                save_and_exit(model, optimizer, scheduler, iters, params, args, world_rank)
            
            # Optional: Log time statistics
            if world_rank == 0 and iters % params.logging_freq == 0:
                elapsed_time = time.time() - start_time
                remaining_time = get_remaining_time()
                hours_remaining = remaining_time / 3600
                logging.info(f"Time elapsed: {elapsed_time:.2f}s, Remaining: {hours_remaining:.2f}h")
                logging.info(f"Current iteration: {iters}/{params.num_iters} ({(iters/params.num_iters)*100:.1f}%)")

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        end = time.time()

        if world_rank == 0:
            iters_per_sec = step_count / (end - start)
            samples_per_sec = params["global_batch_size"] * iters_per_sec
            logging.info(f'Epoch {epoch} | {iters}/{params.num_iters}')
            logging.info(
                "Time taken for epoch %i is %f sec, avg %f samples/sec",
                epoch + 1,
                end - start,
                samples_per_sec,
            )
            logging.info("  Avg train loss=%f" % np.mean(tr_loss))
            args.tboard_writer.add_scalar("Loss/train", np.mean(tr_loss), iters)
            args.tboard_writer.add_scalar(
                "Learning Rate", optimizer.param_groups[0]["lr"], iters
            )
            args.tboard_writer.add_scalar("Avg iters per sec", iters_per_sec, iters)
            args.tboard_writer.add_scalar("Avg samples per sec", samples_per_sec, iters)
            fig = generate_images([inp, tar, gen])
            args.tboard_writer.add_figure("Visualization, t2m", fig, iters, close=True)


        val_start = time.time()
        val_loss = torch.zeros(1, device=device)
        val_rmse = torch.zeros(
            (params.n_out_channels), dtype=torch.float32, device=device
        )
        valid_steps = 0
        model.eval()

        # Validation
        with torch.inference_mode():
            with torch.no_grad():
                for i, data in enumerate(val_data_loader, 0):
                    with autocast(enabled=params.amp_enabled, dtype=params.amp_dtype):
                        inp, tar = map(lambda x: x.to(device), data)
                        gen = model(inp)
                        val_loss += loss_func(gen, tar)
                        val_rmse += weighted_rmse(gen, tar)
                    valid_steps += 1

                if params.distributed:
                    torch.distributed.all_reduce(
                        val_loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                    )
                    torch.distributed.all_reduce(
                        val_rmse, op=ReduceOp.AVG, group=comm.get_group("dp")
                    )

        val_rmse /= valid_steps  # Avg validation rmse
        val_loss /= valid_steps
        val_end = time.time()
        if world_rank == 0:
            val_iters_per_sec = valid_steps / (val_end - val_start)
            val_samples_per_sec = params["global_batch_size"] * iters_per_sec
            logging.info("  Avg val loss={}".format(val_loss.item()))
            logging.info("  Total validation time: {} sec".format(val_end - val_start))
            args.tboard_writer.add_scalar("Loss/valid", val_loss, iters)
            args.tboard_writer.add_scalar("Avg val iters per sec", val_iters_per_sec, iters)
            args.tboard_writer.add_scalar("Avg val samples per sec", val_samples_per_sec, iters)
            args.tboard_writer.add_scalar(
                "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], iters
            )
            args.tboard_writer.flush()
        if iters >= params.num_iters:
            break
            
    torch.cuda.synchronize()
    t2 = time.time()
    tottime = t2 - t1
    pynvml.nvmlShutdown()

def save_checkpoint(model, optimizer, scheduler, iters, params, args, world_rank):
    """Save training checkpoint with model parallel support"""
    if world_rank == 0:
        # Save model configuration and training state
        checkpoint = {
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'iters': iters,
            # Save model parallel configuration
            'parallel_config': {
                'tp_size': params.get('tp', 1),
                'cp_size': params.get('cp', 1),
                'parallel_order': params.get('order', 'tp-cp-dp'),
            },
            # Save model architecture config
            'model_config': {
                'embed_dim': params.embed_dim,
                'depth': params.depth,
                'num_heads': params.num_heads,
                'patch_size': params.patch_size,
            },
            # Save training config
            'training_config': {
                'amp_dtype': str(params.amp_dtype),
                'global_batch_size': params.global_batch_size,
                'local_batch_size': params.local_batch_size,
            }
        }
        
        # Save to temporary file first
        temp_checkpoint_path = os.path.join(params.experiment_dir, f'checkpoint_{iters}.pt.tmp')
        checkpoint_path = os.path.join(params.experiment_dir, f'checkpoint_{iters}.pt')
        torch.save(checkpoint, temp_checkpoint_path)
        # Atomic rename to avoid corrupted checkpoints
        os.rename(temp_checkpoint_path, checkpoint_path)
        
        # Save latest checkpoint symlink
        latest_path = os.path.join(params.experiment_dir, 'checkpoint_latest.pt')
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(f'checkpoint_{iters}.pt', latest_path)
        
        logging.info(f"Saved checkpoint at iteration {iters} to {checkpoint_path}")
        
        # Cleanup old checkpoints if needed
        if hasattr(params, 'keep_n_checkpoints'):
            try:
                checkpoint_files = sorted([
                    f for f in os.listdir(params.experiment_dir) 
                    if f.startswith('checkpoint_') and f.endswith('.pt') and not f == 'checkpoint_latest.pt'
                ])
                for old_ckpt in checkpoint_files[:-params.keep_n_checkpoints]:
                    try:
                        os.remove(os.path.join(params.experiment_dir, old_ckpt))
                        logging.info(f"Removed old checkpoint: {old_ckpt}")
                    except OSError as e:
                        logging.warning(f"Failed to remove checkpoint {old_ckpt}: {e}")
            except Exception as e:
                logging.warning(f"Error during checkpoint cleanup: {e}")

def validate_checkpoint_config(checkpoint, params, world_rank):
    """Validate checkpoint configuration matches current setup"""
    if world_rank == 0:
        # Check parallel configuration
        ckpt_tp = checkpoint['parallel_config']['tp_size']
        ckpt_cp = checkpoint['parallel_config']['cp_size']
        current_tp = params.get('tp', 1)
        current_cp = params.get('cp', 1)
        
        if ckpt_tp != current_tp or ckpt_cp != current_cp:
            raise ValueError(
                f"Checkpoint parallel config (TP={ckpt_tp}, CP={ckpt_cp}) "
                f"doesn't match current config (TP={current_tp}, CP={current_cp})"
            )
            
        # Check model architecture
        for key in ['embed_dim', 'depth', 'num_heads', 'patch_size']:
            ckpt_val = checkpoint['model_config'][key]
            current_val = getattr(params, key)
            if ckpt_val != current_val:
                raise ValueError(
                    f"Checkpoint model config '{key}' ({ckpt_val}) "
                    f"doesn't match current config ({current_val})"
                )
        
        logging.info("Checkpoint configuration validated successfully")

def load_checkpoint(model, optimizer, scheduler, params, args, world_rank, local_rank):
    """Load training checkpoint with model parallel support"""
    # Support loading from iteration number or 'latest'
    if args.resume_iter == -1:
        checkpoint_path = os.path.join(params.experiment_dir, 'checkpoint_latest.pt')
    else:
        checkpoint_path = os.path.join(params.experiment_dir, f'checkpoint_{args.resume_iter}.pt')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
        
        # Validate configurations
        validate_checkpoint_config(checkpoint, params, world_rank)
        
        # Load model weights with appropriate parallel configuration
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # Sync shared weights for context parallel
        if comm.get_size("cp") > 1:
            init_params_for_shared_weights(model)
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        iters = checkpoint['iters']
        
        if world_rank == 0:
            logging.info(f"Loaded checkpoint from iteration {iters}")
            logging.info(f"Training config from checkpoint: {checkpoint['training_config']}")
        
        return iters
    else:
        if world_rank == 0:
            logging.warning(f"No checkpoint found at {checkpoint_path}")
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str, help="tag for indexing the current experiment",)
    parser.add_argument("--scale_depth", type=int, default=1.0, help="Scaling factor for number of transformer layers")
    parser.add_argument("--scale_heads", type=int, default=1.0, help="Scaling factor for number of attention heads")
    parser.add_argument("--scale_dim", type=int, default=1.0, help="Scaling factor for embedding dimension")
    parser.add_argument("--n_train", type=int, default=1.0, help="Number of years to use for training")
    parser.add_argument("--exp_name", type=str, default='default', help="Experiment name")
    parser.add_argument("--budget", default="0.0", type=float, help="Compute budget in FLOPS",)
    parser.add_argument("--yaml_config", default="./config/ViT.yaml", type=str, help="path to yaml file containing training configs")
    parser.add_argument("--config", default="base", type=str, help="name of desired config in yaml file")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="select automatic mixed precision mode")
    parser.add_argument("--enable_fused", action="store_true", help="enable fused Adam optimizer")
    parser.add_argument("--enable_jit", action="store_true", help="enable JIT compilation")
    parser.add_argument("--local_batch_size", default=None, type=int, help="local batchsize (manually override global_batch_size config setting)",)
    parser.add_argument("--num_iters", default=None, type=int, help="number of iters to run")
    parser.add_argument("--num_data_workers", default=None, type=int, help="number of data workers for data loader",)
    parser.add_argument("--data_loader_config", default=None, type=str, choices=["pytorch", "dali"], help="dataloader configuration. choices: 'pytorch', 'dali'")
    parser.add_argument("--bucket_cap_mb", default=25, type=int, help="max message bucket size in mb")
    parser.add_argument("--disable_broadcast_buffers", action="store_true", help="disable syncing broadcasting buffers",)
    parser.add_argument("--noddp", action="store_true", help="disable DDP communication")
    parser.add_argument("--n_nodes", default=4, type=int, help="number of nodes to used (not used here, but logged for later analysis)")

    # model parallelism arguments
    parser.add_argument("--tensor_parallel", default=1, type=int, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--context_parallel", default=1, type=int, help="Number of GPUs for context parallelism")
    parser.add_argument("--parallel_order", default="tp-cp-dp", type=str, help="Order of ranks for parallelism",)

    # checkpointing arguments
    parser.add_argument("--resume_iter", type=int, default=0, help="iteration to resume training from (-1 for latest)")
    parser.add_argument("--checkpoint_freq", type=int, default=1000, help="frequency (in iterations) to save checkpoints")

    # time limit arguments
    parser.add_argument("--time_buffer", type=int, default=60, help="buffer time in seconds before SLURM time limit")
    parser.add_argument("--time_limit", type=str, default="00:30:00", help="SLURM time limit (Not used here, but logged for later analysis)")

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    ########
    # Override YAML params for scaling variables
    params.embed_dim = args.scale_dim
    params.depth = args.scale_depth
    params.num_heads = args.scale_heads
    params.n_train = args.n_train
    params.budget = args.budget
    ########

    # Update config with modified args
    # set up amp

    params.update({"amp_mode": args.amp_mode})
        
    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float32

    params.update(
        {
            "amp_enabled": amp_dtype is not torch.float32,
            "amp_dtype": amp_dtype,
            "enable_fused": args.enable_fused,
            "enable_jit": args.enable_jit,
        }
    )

    if args.data_loader_config:
        params.update({"data_loader_config": args.data_loader_config})

    if args.num_iters:
        params.update({"num_iters": args.num_iters})

    if args.num_data_workers:
        params.update({"num_data_workers": args.num_data_workers})

    params.distributed = False

    # setup model parallel sizes
    params["tp"] = args.tensor_parallel
    params["cp"] = args.context_parallel
    params["order"] = args.parallel_order
    # initialize comm
    comm.init(params, verbose=True)

    # get info from comm
    world_size = comm.get_world_size()
    world_rank = comm.get_world_rank()
    local_rank = comm.get_local_rank()
    params.distributed = world_size > 1

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update(
            {"global_batch_size": comm.get_size("dp") * args.local_batch_size}
        )
    else:
        # Compute local batch size based on number of ranks
        params.local_batch_size = int(
            params["global_batch_size"] // comm.get_size("dp")
        )

    # Move assert after batch size calculations
    assert (
        params["global_batch_size"] % comm.get_size("dp") == 0
    ), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('dp')} GPU."

    # for data loader, set the actual number of data shards and id
    params.data_num_shards = comm.get_size("dp")
    params.data_shard_id = comm.get_rank("dp")

    if world_rank == 0:
        # Directory setup
        baseDir = Path(scratch) / 'scaling_logs'
        baseDir.mkdir(exist_ok=True, parents=True)

        existing = [int(x.name) for x in baseDir.iterdir()]
        if existing:
            run_num = str(max(existing)+1).zfill(3)
        else:
            run_num = '000'
        expDir: Path = baseDir / run_num
        expDir.mkdir(exist_ok=True, parents=True)
        params.experiment_dir = os.path.abspath(expDir)

        # Setup data
        clean_up_temp_dirs(params.n_train)
        data_subset(params.n_train)
        params.train_data_path = str(temp_train/str(params.n_train))
        params.valid_data_path = str(temp_val/str(params.n_train))
        
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(str(expDir), "logs/"))
        
        hparams = {
            'embed': args.scale_dim,
            'layers': args.scale_depth,
            'heads': args.scale_heads,
            'train_years': args.n_train,
            'dtype': str(amp_dtype),
            'compute_budget': args.budget,
            'n_nodes': args.n_nodes,
            'time_limit': args.time_limit
        }
        with open(expDir/'hparams.json', "w") as f:
            json.dump(hparams, f)

    train(params, args, local_rank, world_rank, world_size)
    
    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)
    if world_rank == 0:
        clean_up_temp_dirs(params.n_train)
