import sys
import os
import logging
from utils import logging_utils
logging_utils.config_logger()

logging.info(f"MACHINE={os.getenv('MACHINE')}")
if os.getenv("MACHINE") == "frontier":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")
    os.environ["ROCR_VISIBLE_DEVICES"] = os.environ.get("SLURM_LOCALID", "0")
    logging.info(f"Set GPU device environment variables: HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
