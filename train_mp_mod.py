import sys
import os
import time
import numpy as np
import argparse
import pynvml

import torch
import torch.nn as nn
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

def log_scaling_metrics(model, val_rmse, args):
    """Log metrics relevant to model scaling analysis"""
    if not hasattr(args, "tboard_writer"):
        return
        
    # Get parameter count
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Log efficiency metrics
    compute_efficiency = -val_rmse / param_count  # Higher is better
    memory_usage = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    
    args.tboard_writer.add_scalar('Scaling/ComputeEfficiency', compute_efficiency, param_count)
    args.tboard_writer.add_scalar('Scaling/MemoryUsage', memory_usage, param_count)

def log_embedding_metrics(embed_dim, val_loss, val_rmse, args):
    """Log metrics specific to embedding dimension scaling"""
    if not hasattr(args, "tboard_writer"):
        return
        
    # Memory metrics
    mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    mem_per_dim = mem_allocated / embed_dim
    
    # Performance metrics
    forward_time = args.forward_time_avg if hasattr(args, 'forward_time_avg') else 0
    
    # Log metrics vs embedding dimension
    args.tboard_writer.add_scalar('Embedding/ValidationLoss', val_loss, embed_dim)
    args.tboard_writer.add_scalar('Embedding/ValidationRMSE', val_rmse, embed_dim)
    args.tboard_writer.add_scalar('Embedding/MemoryUsage', mem_allocated, embed_dim)
    args.tboard_writer.add_scalar('Embedding/MemoryPerDim', mem_per_dim, embed_dim)
    args.tboard_writer.add_scalar('Embedding/ForwardLatency', forward_time, embed_dim)
    
    # Log efficiency metric (RMSE reduction per memory)
    memory_efficiency = -val_rmse / mem_allocated
    args.tboard_writer.add_scalar('Embedding/MemoryEfficiency', memory_efficiency, embed_dim)


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
        params, params.train_data_path, params.distributed, train=True
    )
    val_data_loader, valid_dataset = get_data_loader_distributed(
        params, params.valid_data_path, params.distributed, train=False
    )
    logging.info("rank %d, data loader initialized" % (world_rank))

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

    if world_rank == 0:
        logging.info(model)
        all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used / (
            1024.0 * 1024.0 * 1024.0
        )
        logging.info(f"Scaffolding memory high watermark: {all_mem_gb} GB.")

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
            # Log model params one time
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            args.tboard_writer.add_scalar('Scaling/Parameters', param_count, 0)
    

    params.num_epochs = params.num_iters // len(train_data_loader)
    iters = 0
    t1 = time.time()
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
            if world_rank == 0:
                if epoch == 3 and i == 0:
                    torch.cuda.profiler.start()
                if epoch == 3 and i == len(train_data_loader) - 1:
                    torch.cuda.profiler.stop()

            torch.cuda.nvtx.range_push(f"step {i}")
            iters += 1
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

        torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        end = time.time()

        if world_rank == 0:
            iters_per_sec = step_count / (end - start)
            samples_per_sec = params["global_batch_size"] * iters_per_sec
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
            logging.info("  Avg val loss={}".format(val_loss.item()))
            logging.info("  Total validation time: {} sec".format(val_end - val_start))
            args.tboard_writer.add_scalar("Loss/valid", val_loss, iters)
            args.tboard_writer.add_scalar(
                "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], iters
            )
            log_scaling_metrics(model, val_rmse.cpu().numpy()[0], args)
            log_embedding_metrics(params.embed_dim, val_loss, val_rmse.cpu().numpy()[0], args)
            args.tboard_writer.flush()
            
    torch.cuda.synchronize()
    t2 = time.time()
    tottime = t2 - t1
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_num",
        default="00",
        type=str,
        help="tag for indexing the current experiment",
    )
    ########################
    # Add to existing ArgumentParser
    parser.add_argument(
        "--scale_depth",
        type=int,
        default=1.0,
        help="Scaling factor for number of transformer layers"
    )
    parser.add_argument(
        "--scale_heads",
        type=int,
        default=1.0,
        help="Scaling factor for number of attention heads"
    )
    parser.add_argument(
        "--scale_dim",
        type=int,
        default=1.0,
        help="Scaling factor for embedding dimension"
    )
    #########################
    parser.add_argument(
        "--yaml_config",
        default="./config/ViT.yaml",
        type=str,
        help="path to yaml file containing training configs",
    )
    parser.add_argument(
        "--config", default="base", type=str, help="name of desired config in yaml file"
    )
    parser.add_argument(
        "--amp_mode",
        default="none",
        type=str,
        choices=["none", "fp16", "bf16"],
        help="select automatic mixed precision mode",
    )
    parser.add_argument(
        "--enable_fused", action="store_true", help="enable fused Adam optimizer"
    )
    parser.add_argument(
        "--enable_jit", action="store_true", help="enable JIT compilation"
    )
    parser.add_argument(
        "--local_batch_size",
        default=None,
        type=int,
        help="local batchsize (manually override global_batch_size config setting)",
    )
    parser.add_argument(
        "--num_iters", default=None, type=int, help="number of iters to run"
    )
    parser.add_argument(
        "--num_data_workers",
        default=None,
        type=int,
        help="number of data workers for data loader",
    )
    parser.add_argument(
        "--data_loader_config",
        default=None,
        type=str,
        choices=["pytorch", "dali"],
        help="dataloader configuration. choices: 'pytorch', 'dali'",
    )
    parser.add_argument(
        "--bucket_cap_mb", default=25, type=int, help="max message bucket size in mb"
    )
    parser.add_argument(
        "--disable_broadcast_buffers",
        action="store_true",
        help="disable syncing broadcasting buffers",
    )
    parser.add_argument(
        "--noddp", action="store_true", help="disable DDP communication"
    )

    # model parallelism arguments
    parser.add_argument(
        "--tensor_parallel",
        default=1,
        type=int,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--context_parallel",
        default=1,
        type=int,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--parallel_order",
        default="tp-cp-dp",
        type=str,
        help="Order of ranks for parallelism",
    )

    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    ########
    # Override YAML params for scaling variables
    params.embed_dim = args.scale_dim
    params.depth = args.scale_depth
    params.num_heads = args.scale_heads
    ########
    param_str = f"L{args.scale_depth}_H{args.scale_heads}"

    # Update config with modified args
    # set up amp
    if args.amp_mode != "none":
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32
    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16
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

    assert (
        params["global_batch_size"] % comm.get_size("dp") == 0
    ), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('dp')} GPU."

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

    # for data loader, set the actual number of data shards and id
    params.data_num_shards = comm.get_size("dp")
    params.data_shard_id = comm.get_rank("dp")

    # Set up directory
    baseDir = params.expdir
    # expDir = os.path.join(
    #     baseDir, args.config + "/%dMP/" % (comm.get_size("tp-cp")) + str(run_num) + f"_{param_str}" +"/"
    # )
    expDir = os.path.join(
        baseDir, args.config + "/%dMP/" % (comm.get_size("tp-cp")) + str(run_num) + f"_emb{args.scale_dim}" +"/"
    )
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, "logs/"))

    params.experiment_dir = os.path.abspath(expDir)

    train(params, args, local_rank, world_rank, world_size)

    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)