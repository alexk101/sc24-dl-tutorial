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

scratch = os.getenv("SCRATCH")
temp_train = Path(f"{scratch}/temp_train")
temp_val = Path(f"{scratch}/temp_val")

def data_subset(n_train: int=25):
    target = Path('/pscratch/sd/s/shas1693/data/sc24_tutorial_data')
    all_data = list((target/'train').iterdir())
    all_data += list((target/'valid').iterdir())
    sorted(all_data)
    train_subset = all_data[:n_train]
    valid_subset = all_data[n_train:]

    (temp_train/str(n_train)).mkdir(exist_ok=True, parents=True)
    (temp_val/str(n_train)).mkdir(exist_ok=True, parents=True)

    for x in train_subset:
        os.symlink(x, temp_train/str(n_train)/x.name)
    
    for x in valid_subset:
        os.symlink(x, temp_val/str(n_train)/x.name)


def clean_up_temp_dirs(n_train: int):
    for x in (temp_train/str(n_train)).iterdir():
        os.unlink(x)

    for x in (temp_val/str(n_train)).iterdir():
        os.unlink(x)
    os.unlink((temp_val/str(n_train)))
    os.unlink((temp_train/str(n_train)))


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
    train_data_loader = get_data_loader_distributed(params, params.train_data_path, params.distributed, train=True)
    val_data_loader = get_data_loader_distributed(params, params.valid_data_path, params.distributed, train=False)
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

    # Calculate iterations for budget
    if params.budget:
        # Calculate sequence length
        seq_len = (360 // params.patch_size) * (720 // params.patch_size)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Number of iterations to run based on desired flops
        tokens_per_step = params.global_batch_size * seq_len
        max_steps = int(params.budget // (6 * param_count * tokens_per_step))
        params.num_iters = max_steps // (params.global_batch_size * seq_len)

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

    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95))

    if world_rank == 0:
        logging.info(model)

    # Track iterations globally across distributed ranks
    global_step = 0

    # Training loop with step-based control
    model.train()
    while global_step < params.num_iters:
        for batch_idx, (inp, tar) in enumerate(train_data_loader):
            if global_step >= params.num_iters:
                break

            inp, tar = inp.to(device), tar.to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=params.amp_enabled, dtype=params.amp_dtype):
                gen = model(inp)
                loss = l2_loss(gen, tar)

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
            # lr step
            scheduler.step()

            global_step += 1

            # Log training progress
            if world_rank == 0 and global_step % 100 == 0:
                logging.info(f"Step {global_step}/{params.num_iters}, Loss: {loss.item():.4f}")
                # Log memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            logging.info(f"Rank {world_rank}, Step {global_step}: Memory Used: {mem_info.used / (1024 ** 3):.2f} GB, Free: {mem_info.free / (1024 ** 3):.2f} GB")

    # Shutdown pynvml
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str, help="tag for indexing the current experiment")
    parser.add_argument("--scale_depth", type=int, default=1.0, help="Scaling factor for number of transformer layers")
    parser.add_argument("--scale_heads", type=int, default=1.0, help="Scaling factor for number of attention heads")
    parser.add_argument("--scale_dim", type=int, default=1.0, help="Scaling factor for embedding dimension")
    parser.add_argument("--yaml_config", default="./config/ViT.yaml", type=str, help="path to yaml file containing training configs")
    parser.add_argument("--config", default="base", type=str, help="name of desired config in yaml file")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "fp16", "bf16"], help="select automatic mixed precision mode")
    parser.add_argument("--enable_fused", action="store_true", help="enable fused Adam optimizer")
    parser.add_argument("--enable_jit", action="store_true", help="enable JIT compilation")
    parser.add_argument("--num_iters", default=None, type=int, help="number of iterations to run")
    parser.add_argument("--budget", default=None, type=int, help="compute budget in flops")

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    
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

    # Directory setup
    baseDir = Path(params.expdir)
    existing = [int(x.name) for x in baseDir.iterdir()]
    run_num = str(max(existing)).zfill()
    expDir: Path = baseDir / run_num

    # Hyperparams
    params.embed_dim = args.scale_dim
    params.depth = args.scale_depth
    params.num_heads = args.scale_heads
    params.n_train = args.n_train
    params.budget = args.budget
    params.num_iters = args.num_iters

    # for data loader, set the actual number of data shards and id
    params.data_num_shards = comm.get_size("dp")
    params.data_shard_id = comm.get_rank("dp")

    hparams = {
        'embed': args.scale_dim,
        'layers': args.scale_depth,
        'heads': args.scale_heads,
        'train_years': args.n_train,
        'dtype': str(amp_dtype),
        'compute_budget': args.budget
    }
    args.tboard_writer.add_hparams(hparams)

    if world_rank == 0:
        # Setup data
        data_subset(params.n_train)
        expDir.mkdir(exist_ok=True, parents=True)
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        params.log()
        args.tboard_writer = SummaryWriter(log_dir=os.path.join(str(expDir), "logs/"))

    train(params, args, local_rank, world_rank, world_size)

    if world_rank == 0:
        clean_up_temp_dirs(params.n_train)