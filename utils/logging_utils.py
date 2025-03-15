import os
import logging
import sys
from pathlib import Path
# Custom logger that includes rank
_format = '%(asctime)s - Rank %(rank)s - %(message)s'

class RankLogger:
  def __init__(self, rank):
    self.rank = rank
    self.logger = logging.getLogger()
    self.log_level = logging.INFO
    rank_log_dir = Path(__file__).parent.parent / f"logs_{os.environ.get('SLURM_JOB_ID', '?')}"
    rank_log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
      level=self.log_level,
      format=_format,
      handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(rank_log_dir / f"dist_test_rank{os.environ.get('SLURM_PROCID', '?')}.log"))
      ]
    )
        
  def info(self, msg):
    self.logger.info(msg, extra={'rank': self.rank})
      
  def error(self, msg):
    self.logger.error(msg, extra={'rank': self.rank})

  def warning(self, msg):
    self.logger.warning(msg, extra={'rank': self.rank})

  def debug(self, msg):
    self.logger.debug(msg, extra={'rank': self.rank})

  def log_to_file(self, log_filename='tensorflow.log'):
 
    if not os.path.exists(os.path.dirname(log_filename)):
      os.makedirs(os.path.dirname(log_filename))

    fh = logging.FileHandler(log_filename)
    fh.setLevel(self.log_level)
    fh.setFormatter(logging.Formatter(_format))
    self.logger.addHandler(fh)

GLOBAL_LOG = RankLogger(os.environ.get("SLURM_PROCID", "?"))