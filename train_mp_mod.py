import sys
import os
import logging
if os.getenv("MACHINE") == "frontier":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")
    os.environ["ROCR_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")
    logging.info(f"Set GPU device environment variables: HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")

import time
import numpy as np
import argparse

from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams

# Now import the rest of the modules
from utils import get_data_loader_distributed
from utils import comm
from utils.loss import l2_loss, l2_loss_opt
from utils.metrics import weighted_rmse, time_communication, backward_with_comm_timing
from utils.data import data_subset, clean_up_temp_dirs, TEMP_TRAIN, TEMP_VAL, SCRATCH
from networks import vit

from distributed.mappings import init_ddp_model_and_reduction_hooks
from distributed.helpers import init_params_for_shared_weights

from utils.plots import generate_images, calculate_layerwise_stdv, save_stdv_values, log_stdv_image
from pathlib import Path
import json

# Vendor-agnostic PyTorch imports
import torch.optim as optim
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp, destroy_process_group
from torch.amp import autocast, GradScaler

# GPU vendor-specific imports
import torch
from utils.gpu_utils import (
    NVIDIA_AVAILABLE, ROCM_AVAILABLE, GPU_BACKEND, 
    get_gpu_info, initialize_gpu, get_profiler, log_rocm_utilization
)


# Check for bfloat16 support
BFLOAT16_AVAILABLE = False
device_type = None
if NVIDIA_AVAILABLE or ROCM_AVAILABLE:
    BFLOAT16_AVAILABLE = torch.cuda.is_bf16_supported()
    device_type = "cuda"
    # from torch.cuda.amp import autocast, GradScaler
    logging.info(f"bfloat16 support: {BFLOAT16_AVAILABLE}")
else:
    raise RuntimeError("No GPU support available. This script requires either NVIDIA CUDA or AMD ROCm GPUs.")

from torch.utils.flop_counter import FlopCounterMode

def validate_amp_dtype(requested_dtype):
    """Validate and adjust AMP dtype based on hardware support"""
    if requested_dtype == torch.bfloat16 and not BFLOAT16_AVAILABLE:
        logging.warning("BFloat16 requested but not supported by hardware. Falling back to FP16")
        return torch.float16
    return requested_dtype


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
            destroy_process_group(None)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error during save_and_exit: {e}")
        sys.exit(1)

# Get profiler once at module level
profiler = get_profiler()

def validate_model(model, val_loader, device, params, loss_func, world_rank, comm=None):
    model.eval()
    val_loss = torch.zeros(1, device=device)
    val_rmse = torch.zeros((params.n_out_channels), dtype=torch.float32, device=device)
    valid_steps = 0
    
    with torch.inference_mode():
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                with autocast(device_type=device_type, enabled=params.amp_enabled, dtype=params.amp_dtype):
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

    val_rmse /= valid_steps
    val_loss /= valid_steps
    
    return val_loss, val_rmse, valid_steps

def train(params, args, local_rank, world_rank, world_size, hyperparameter_search=False):
    # Initialize tracking variables for hyperparameter search
    if hyperparameter_search:
        best_val_rmse = float('inf')
        patience_counter = 0
        early_stop_patience = 5
        val_freq = 100
        training_start_time = time.time()
        peak_memory = 0
    
    # Initialize GPU and get device handle
    device, gpu_handle = initialize_gpu(local_rank)

    # Validate and adjust AMP dtype based on hardware support
    if hasattr(params, 'amp_dtype'):
        params.amp_dtype = validate_amp_dtype(params.amp_dtype)
        if world_rank == 0:
            logging.info(f"Using AMP dtype: {params.amp_dtype}")

    # Get data loader
    logging.info("rank %d, begin data loader init" % world_rank)
    
    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(
        params, str(TEMP_TRAIN/str(params.n_train)), params.distributed, train=True
    )
    val_data_loader, valid_dataset = get_data_loader_distributed(
        params, str(TEMP_VAL/str(params.n_train)), params.distributed, train=False
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

    # Log GPU details
    gpu_info = get_gpu_info(local_rank)
    logging.info(
        f"Rank {world_rank}: Using {gpu_info['name']}, "
        f"Total GPU memory: {gpu_info['total_memory']/(1024**3):.2f} GB"
    )

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
        scaler = GradScaler(device_type=device_type)

    # weight initialization needs to be synced across shared weights
    if comm.get_size("tp-cp") > 1:
        logging.info("Init shared weights")
        init_params_for_shared_weights(model)

    if params.distributed and not args.noddp:
        logging.info("Init DDP")
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
        gpu_info = get_gpu_info(local_rank)
        all_mem_gb = gpu_info['used_memory'] / (1024.0**3)
        logging.info(f"Scaffolding memory high watermark: {all_mem_gb:.2f} GB.")
        with open(f'{params.experiment_dir}/hparams.json', 'r') as file:
            data = json.load(file)
        data['parameters'] = param_count
        with open(f'{params.experiment_dir}/hparams.json', 'w') as file:
            json.dump(data, file, indent=4)

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
        val_loss, val_rmse, valid_steps = validate_model(model, val_data_loader, device, params, loss_func, world_rank, comm)
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
    
    # Set time check frequency (e.g., every 100 iterations)
    time_check_freq = 100  # Can be adjusted based on your needs
    if world_rank == 0:
        logging.info(f"Will check remaining time every {time_check_freq} iterations")

    # Get initial FLOP count with a sample input
    def count_training_flops(model, sample_input, loss_func):
        flop_counter = FlopCounterMode()
        with flop_counter:
            with autocast(device_type=device_type, enabled=params.amp_enabled, dtype=params.amp_dtype):
                output = model(sample_input)
                loss = loss_func(output, sample_input)  # Using input as dummy target
            loss.backward()
        return flop_counter.get_total_flops()

    sample_input = next(iter(train_data_loader))[0].to(device)
    model.train()
    flops_per_step = count_training_flops(model, sample_input, loss_func)
    total_flops = 0

    if world_rank == 0:
        logging.info(f"FLOPs per training step: {flops_per_step:,}")

    # Training loop
    for epoch in range(startEpoch, startEpoch + params.num_epochs):
        if world_rank == 0:
            logging.info(f"About to synchronize at epoch {epoch} start")
        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        if world_rank == 0:
            logging.info(f"Synchronized at epoch {epoch} start")
        if params.distributed and (train_sampler is not None):
            train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []
        tr_time = 0.0
        dat_time = 0.0
        log_time = 0.0

        if world_rank == 0:
            logging.info(f"Training loop started at epoch {epoch}")
        model.train()
        step_count = 0

        for i, data in enumerate(train_data_loader, 0):
            if iters >= params.num_iters:
                if world_rank == 0:
                    logging.info("Reached maximum iterations, initiating shutdown...")
                save_and_exit(model, optimizer, scheduler, iters, params, args, world_rank)
                
            if world_rank == 0:
                if epoch == 3 and i == 0:
                    torch.cuda.profiler.start()
                if epoch == 3 and i == len(train_data_loader) - 1:
                    torch.cuda.profiler.stop()

            if profiler:
                profiler.range_push(f"step {i}")
            
            dat_start = time.time()
            if profiler:
                profiler.range_push(f"data copy in {i}")

            inp, tar = map(lambda x: x.to(device), data)
            if profiler:
                profiler.range_pop()  # copy in

            tr_start = time.time()
            b_size = inp.size(0)

            optimizer.zero_grad()

            if profiler:
                profiler.range_push(f"forward")
            with autocast(device_type=device_type, enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)
            if profiler:
                profiler.range_pop()  # forward

            if world_rank == 0 and i == 1:  # print the mem used
                gpu_info = get_gpu_info(local_rank)
                all_mem_gb = gpu_info['used_memory'] / (1024.0 * 1024.0 * 1024.0)
                logging.info(f" Memory usage after forward pass: {all_mem_gb} GB.")

            if params.amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                if profiler:
                    profiler.range_push(f"optimizer")
                scaler.step(optimizer)
                if profiler:
                    profiler.range_pop()  # optimizer
                scaler.update()
            else:
                # Replace with timing instrumentation
                timing_stats = backward_with_comm_timing(loss, optimizer)
                if world_rank == 0 and iters % params.logging_freq == 0:
                    logging.info(f"Backward timing: compute={timing_stats['backward_compute_time']:.4f}s, "
                                 f"comm={timing_stats['comm_time']:.4f}s, "
                                 f"ratio={timing_stats['comm_ratio']:.2%}")
                    args.tboard_writer.add_scalar("Performance/comm_ratio", timing_stats["comm_ratio"], iters)

            if params.distributed:
                logging.info(f"Rank {world_rank}: About to perform all_reduce operation")
                torch.distributed.all_reduce(
                    loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                )
                logging.info(f"Rank {world_rank}: All_reduce operation completed")
            tr_loss.append(loss.item())

            if profiler:
                profiler.range_pop()  # step
            # lr step
            scheduler.step()

            tr_end = time.time()
            tr_time += tr_end - tr_start
            dat_time += tr_start - dat_start
            step_count += 1
            iters += 1

            if hyperparameter_search and (iters % val_freq == 0):
                val_loss, val_rmse, _ = validate_model(
                    model, val_data_loader, device, params,
                    loss_func, world_rank, comm if params.distributed else None
                )
                
                gpu_info = get_gpu_info(local_rank)
                current_memory = gpu_info['used_memory'] / (1024.0 * 1024.0 * 1024.0)
                peak_memory = max(peak_memory, current_memory)
                
                if val_rmse.cpu().numpy()[0] < best_val_rmse:
                    best_val_rmse = val_rmse.cpu().numpy()[0]
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stop_patience:
                    if world_rank == 0:
                        logging.info(f"Early stopping triggered at iteration {iters}")
                    training_time = time.time() - training_start_time
                    return best_val_rmse, peak_memory, training_time

            # Check remaining time periodically
            if iters % time_check_freq == 0:
                if world_rank == 0:
                    remaining_time = torch.tensor(get_remaining_time(), device=device)
                else:
                    remaining_time = torch.tensor(0.0, device=device)
                    
                if params.distributed:
                    torch.distributed.broadcast(remaining_time, src=0)
                
                if remaining_time.item() < time_buffer:
                    if world_rank == 0:
                        logging.info(f"Time limit approaching (remaining: {remaining_time.item():.1f}s)")
                    save_and_exit(model, optimizer, scheduler, iters, params, args, world_rank)

            # Optional: Log time and FLOP statistics
            if iters % params.logging_freq == 0:
                comm_stats = time_communication(comm, device)
                if world_rank == 0:
                    logging.info(f"Communication stats: {comm_stats}")
                    args.tboard_writer.add_scalar("Comm/all_reduce_time_ms", comm_stats["all_reduce_time_ms"], iters)
                    args.tboard_writer.add_scalar("Comm/broadcast_time_ms", comm_stats["broadcast_time_ms"], iters)
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
            log_rocm_utilization()


        val_start = time.time()
        val_loss, val_rmse, valid_steps = validate_model(
            model, val_data_loader, device, params, 
            loss_func, world_rank, comm if params.distributed else None
        )
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
            
            # Add this section to calculate and log standard deviation of the error
            with torch.no_grad():
                # Get a batch of validation data
                inp_val, tar_val = next(iter(val_data_loader))
                inp_val, tar_val = inp_val.to(device), tar_val.to(device)
                
                # Generate predictions
                with autocast(device_type=device_type, enabled=params.amp_enabled, dtype=params.amp_dtype):
                    gen_val = model(inp_val)
                
                # Calculate the error (prediction - target)
                error = gen_val - tar_val
                
                # Calculate layer-wise standard deviation of the error
                error_stdv = calculate_layerwise_stdv(error, tag="error_stdv")
                
                # Save the raw error stdv values for post-processing
                save_dir = os.path.join(params.experiment_dir, "error_stdv_values")
                save_stdv_values(error_stdv, save_dir, iters, "error_stdv")
                
                # Log the average standard deviation of error as an image
                log_stdv_image(error_stdv, args.tboard_writer, global_step=iters, tag="error_stdv_image", save_dir=save_dir)
            # Calculate elapsed time from the start of training
            elapsed_time = time.time() - start_time
            total_flops += flops_per_step
            flops_per_second = total_flops / elapsed_time
            logging.info(f"Total FLOPs: {total_flops:,}")
            logging.info(f"FLOPS/second: {flops_per_second:,.2f}")
            args.tboard_writer.add_scalar('Performance/total_flops', total_flops, iters)
            args.tboard_writer.add_scalar('Performance/flops_per_second', flops_per_second, iters)
            args.tboard_writer.flush()
        if iters >= params.num_iters:
            break
            
    torch.cuda.synchronize()
    t2 = time.time()
    tottime = t2 - t1
    if hyperparameter_search:
        training_time = time.time() - training_start_time
        return best_val_rmse, peak_memory, training_time

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

    # Add this with the other arguments
    parser.add_argument("--learning_rate", type=float, default=None, help="Override the default learning rate")

    # Add with other scaling arguments
    parser.add_argument("--patch_size", type=int, default=None, help="Override the default patch size")

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    ########
    # Override YAML params for scaling variables
    params.embed_dim = args.scale_dim
    params.depth = args.scale_depth
    params.num_heads = args.scale_heads
    params.n_train = args.n_train
    if args.learning_rate is not None:
        params.lr = args.learning_rate
    if args.patch_size is not None:
        params.patch_size = args.patch_size
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
    logging.info(f"Post init: World size: {world_size}")
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
        baseDir = Path(SCRATCH) / 'scaling_logs'
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
        data_subset(params.n_train)
        params.train_data_path = str(TEMP_TRAIN/str(params.n_train))
        params.valid_data_path = str(TEMP_VAL/str(params.n_train))
        
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
            'n_nodes': args.n_nodes,
            'time_limit': args.time_limit,
            'local_batch_size': args.local_batch_size,
            'learning_rate': args.learning_rate,
            'patch_size': args.patch_size,
        }
        with open(expDir/'hparams.json', "w") as f:
            json.dump(hparams, f)

    logging.info(f"Rank {world_rank}: MASTER_PORT={os.environ.get('MASTER_PORT')}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}")

    train(params, args, local_rank, world_rank, world_size)
    
    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)