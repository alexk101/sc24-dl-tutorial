import os
from pathlib import Path
import logging

DATA_DIR = os.getenv("DATADIR")
SCRATCH = os.getenv("SCRATCH")
TEMP_TRAIN = Path(f"{SCRATCH}/temp_train")
TEMP_VAL = Path(f"{SCRATCH}/temp_val")

def data_subset(n_train: int=25):
    if (TEMP_TRAIN/str(n_train)).exists() and (TEMP_VAL/str(n_train)).exists():
        logging.info(f"Temp train and val dirs {TEMP_TRAIN/str(n_train)} and {TEMP_VAL/str(n_train)} already exist")
        return

    target = Path(DATA_DIR)
    all_data = list((target/'train').iterdir())
    all_data = sorted(all_data)
    train_subset = all_data[:n_train]

    (TEMP_TRAIN/str(n_train)).mkdir(exist_ok=True, parents=True)
    (TEMP_VAL/str(n_train)).mkdir(exist_ok=True, parents=True)

    for x in train_subset:
        if not (TEMP_TRAIN/str(n_train)/x.name).exists():
            os.symlink(x, TEMP_TRAIN/str(n_train)/x.name)
    
    for x in (target/'valid').iterdir():
        if not (TEMP_VAL/str(n_train)/x.name).exists():
            os.symlink(x, TEMP_VAL/str(n_train)/x.name)


def clean_up_temp_dirs(n_train: int):
    if (TEMP_TRAIN/str(n_train)).exists():
        for x in (TEMP_TRAIN/str(n_train)).iterdir():
            os.unlink(x)
        (TEMP_TRAIN/str(n_train)).rmdir()
    else:
        logging.info(f"Temp train dir {TEMP_TRAIN/str(n_train)} does not exist")

    if (TEMP_VAL/str(n_train)).exists():
        for x in (TEMP_VAL/str(n_train)).iterdir():
            os.unlink(x)
        (TEMP_VAL/str(n_train)).rmdir()
    else:
        logging.info(f"Temp val dir {TEMP_VAL/str(n_train)} does not exist")
    