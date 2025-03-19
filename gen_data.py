from utils.data import data_subset
import os

os.environ["DATA_DIR"] = "/lustre/orion/geo163/proj-shared/downsampled_data"
os.environ["SCRATCH"] = "/lustre/orion/geo163/scratch/kiefera"

subsets = [1, 5, 10, 15, 20, 25]
for subset in subsets:
    data_subset(subset)
