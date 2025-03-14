import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Rank %(rank)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"dist_test_rank{os.environ.get('SLURM_PROCID', '?')}.log")
    ]
)
# Custom logger that includes rank
class RankLogger:
    def __init__(self, rank):
        self.rank = rank
        self.logger = logging.getLogger()
        
    def info(self, msg):
        self.logger.info(msg, extra={'rank': self.rank})
        
    def error(self, msg):
        self.logger.error(msg, extra={'rank': self.rank})

GLOBAL_LOG = RankLogger(os.environ.get("SLURM_PROCID", "?"))