import os

os.environ["DATADIR"] = "/lustre/orion/geo163/proj-shared/downsampled_data"
os.environ["SCRATCH"] = "/lustre/orion/geo163/scratch/kiefera"

from utils.data import data_subset

subsets = [1, 5, 10, 15, 20, 25]
for subset in subsets:
    data_subset(subset)
