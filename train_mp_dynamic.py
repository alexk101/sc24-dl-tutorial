import sys
import os
import time
import numpy as np
import argparse
import pynvml

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast, GradScaler
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import ReduceOp
from utils.plots import generate_images

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
from pathlib import Path
import json

scratch = os.getenv("SCRATCH")
temp_train = Path(f"{scratch}/temp_train")
temp_val = Path(f"{scratch}/temp_val")

def data_subset(n_train: int=25):
    target = Path('/pscratch/sd/s/shas1693/data/sc24_tutorial_data')
    all_data = list((target/'train').iterdir())
    all_data += list((target/'valid').iterdir())
    all_data = sorted(all_data)
    train_subset = all_data[:n_train]
    valid_subset = all_data[n_train:]

    (temp_train/str(n_train)).mkdir(exist_ok=True, parents=True)
    (temp_val/str(n_train)).mkdir(exist_ok=True, parents=True)

    for x in train_subset:
        if not (temp_train/str(n_train)/x.name).exists():
            os.symlink(x, temp_train/str(n_train)/x.name)
    
    for x in valid_subset:
        if not (temp_val/str(n_train)/x.name).exists():
            os.symlink(x, temp_val/str(n_train)/x.name)


def clean_up_temp_dirs(n_train: int):
    for x in (temp_train/str(n_train)).iterdir():
        os.unlink(x)

    for x in (temp_val/str(n_train)).iterdir():
        os.unlink(x)
    os.unlink((temp_val/str(n_train)))
    os.unlink((temp_train/str(n_train)))


def init_logs(model, device, train_data_loader, val_data_loader, loss_func, args):
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


def log_iter(args, step_count, start, end, epoch, iters, tr_loss, optimizer, inp, tar, gen):
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


def validation(model, device, val_data_loader, loss_func, iters):
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
        logging.info("  Avg val loss={}".format(val_loss.item()))
        logging.info("  Total validation time: {} sec".format(val_end - val_start))
        args.tboard_writer.add_scalar("Loss/valid", val_loss, iters)
        args.tboard_writer.add_scalar(
            "RMSE(u10m)/valid", val_rmse.cpu().numpy()[0], iters
        )
        args.tboard_writer.flush()


def train(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:%d" % local_rank)

    # Initialize pynvml for memory tracking
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)

    # Get data loader
    logging.info("rank %d, begin data loader init" % world_rank)
    train_data_loader, train_dataset, train_sampler = get_data_loader_distributed(params, params.train_data_path, params.distributed, train=True)
    val_data_loader, valid_dataset  = get_data_loader_distributed(params, params.valid_data_path, params.distributed, train=False)
    logging.info("rank %d, data loader initialized" % world_rank)

    # Create model
    model = vit.ViT(params).to(device)

    if params.enable_jit:
        model = torch.compile(model)

    if params.amp_dtype == torch.float16:
        scaler = GradScaler('cuda')

    if comm.get_size("tp-cp") > 1:
        init_params_for_shared_weights(model)

    if params.distributed and not args.noddp:
        model = init_ddp_model_and_reduction_hooks(model, device_ids=[local_rank],
                                                   output_device=[local_rank],
                                                   bucket_cap_mb=args.bucket_cap_mb)

    # select loss function
    if params.enable_jit:
        loss_func = l2_loss_opt
    else:
        loss_func = l2_loss

    # Calculate iterations for budget
    if params.budget:
        # Calculate sequence length
        seq_len = (360 // params.patch_size) * (720 // params.patch_size)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Number of iterations to run based on desired flops
        tokens_per_step = params.global_batch_size * seq_len
        max_steps = int(params.budget // (6 * param_count * tokens_per_step))
        params.num_iters = max_steps // (params.global_batch_size * seq_len)

    params.num_iters = comm.broadcast_value(params.num_iters, src=0)

    if params.enable_fused:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, fused=True, betas=(0.9, 0.95))
    else:
        optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

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

    if world_rank == 0:
        logging.info(model)
        logging.info(f"num_iters: {params.num_iters}")

    init_logs(model, device, train_data_loader, val_data_loader, loss_func, args)
    # Track iterations globally across distributed ranks
    global_step = torch.tensor(0, device=device)

    # Training loop with step-based control
    model.train()
    for epoch in range((params.num_iters // len(train_data_loader)) + 1):
        torch.cuda.synchronize()
        train_sampler.set_epoch(epoch)
        start = time.time()
        tr_loss = []
        step_count = 0

        for batch_idx, (inp, tar) in enumerate(train_data_loader):
            torch.distributed.all_reduce(
                global_step, op=torch.distributed.ReduceOp.MAX, group=comm.get_group("dp")
            )
            if global_step.item() >= params.num_iters:
                break

            inp, tar = inp.to(device), tar.to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = loss_func(gen, tar)

            if params.amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if params.distributed:
                torch.distributed.all_reduce(
                    loss, op=ReduceOp.AVG, group=comm.get_group("dp")
                )
            tr_loss.append(loss.item())
            # lr step
            scheduler.step()

            step_count += 1
            global_step += 1

        end = time.time()
        # Log training progress
        if world_rank == 0 and global_step % 100 == 0:
            logging.info(f"Step {global_step}/{params.num_iters}, Loss: {loss.item():.4f}")
            # Log memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            logging.info(f"Rank {world_rank}, Step {global_step}: Memory Used: {mem_info.used / (1024 ** 3):.2f} GB, Free: {mem_info.free / (1024 ** 3):.2f} GB")
            log_iter(args, step_count, start, end, epoch, global_step, tr_loss, optimizer, inp, tar, gen)
        torch.cuda.synchronize()
        validation(model, device, val_data_loader, loss_func, global_step)
    # Shutdown pynvml
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str, help="tag for indexing the current experiment")
    parser.add_argument("--scale_depth", type=int, default=8, help="Scaling factor for number of transformer layers")
    parser.add_argument("--scale_heads", type=int, default=12, help="Scaling factor for number of attention heads")
    parser.add_argument("--scale_dim", type=int, default=256, help="Scaling factor for embedding dimension")
    parser.add_argument("--yaml_config", default="./config/ViT.yaml", type=str, help="path to yaml file containing training configs")
    parser.add_argument("--config", default="base", type=str, help="name of desired config in yaml file")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="select automatic mixed precision mode")
    parser.add_argument("--enable_fused", action="store_true", help="enable fused Adam optimizer")
    parser.add_argument("--enable_jit", action="store_true", help="enable JIT compilation")
    parser.add_argument("--budget", default=0.0, type=float, help="compute budget in flops")
    parser.add_argument("--n_train", default=25, type=int, help="number of training years")
    parser.add_argument("--local_batch_size", default=None, type=int, help="local batchsize (manually override global_batch_size config setting)",)
    # model parallelism arguments
    parser.add_argument("--tensor_parallel", default=1, type=int, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--context_parallel", default=1, type=int, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--parallel_order", default="tp-cp-dp", type=str, help="Order of ranks for parallelism")
    parser.add_argument("--noddp", action="store_true", help="disable DDP communication")
    parser.add_argument("--bucket_cap_mb", default=25, type=int, help="max message bucket size in mb")
    parser.add_argument("--num_iters", default=None, type=int, help="number of iters to run")

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
    # setup model parallel sizes
    params["tp"] = args.tensor_parallel
    params["cp"] = args.context_parallel
    params["order"] = args.parallel_order
    # initialize comm
    comm.init(params, verbose=True)

    # Setup comm variables
    local_rank = comm.get_local_rank()
    world_rank = comm.get_world_rank()
    world_size = comm.get_world_size()
    params.distributed = world_size > 1

    assert (
        params["global_batch_size"] % comm.get_size("dp") == 0
    ), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('dp')} GPU."


    # Setup amp
    if args.amp_mode != "none":
        params.update({"amp_mode": args.amp_mode})
    amp_dtype = torch.float32
    if params.amp_mode == "fp16":
        amp_dtype = torch.float16
    elif params.amp_mode == "bf16":
        amp_dtype = torch.bfloat16


    # Hyperparams
    params.embed_dim = args.scale_dim
    params.depth = args.scale_depth
    params.num_heads = args.scale_heads
    params.n_train = args.n_train
    params.budget = args.budget
    params.amp_enabled = amp_dtype is not torch.float32
    params.amp_dtype = amp_dtype
    params.enable_fused = args.enable_fused
    params.enable_jit = args.enable_jit

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

        # Setup data
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
            'compute_budget': args.budget
        }
        with open(expDir/'hparams.json', "w") as f:
            json.dump(hparams, f)

    train(params, args, local_rank, world_rank, world_size)

    if world_rank == 0:
        clean_up_temp_dirs(params.n_train)

    if params.distributed:
        torch.distributed.barrier()
    logging.info("DONE ---- rank %d" % world_rank)