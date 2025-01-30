import numpy as np
import dask.array as da
import h5py
import zarr
import os
import json
import torch
from pathlib import Path
from multiprocessing import cpu_count
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader

class TestDataGenerator:
    def __init__(self, mean_file, std_file, output_dir="test_data", format="zarr", 
                 spatial=(100, 100), target_spatial=None, temporal=10, channels=5, distribution="normal", 
                 cache=True, regenerate=False, years=None):
        self.mean = np.load(mean_file)
        self.std = np.load(std_file)
        self.output_dir = Path(output_dir)
        self.format = format.lower()
        self.spatial = spatial
        self.target_spatial = target_spatial if target_spatial else spatial
        self.temporal = temporal
        self.channels = channels
        self.distribution = distribution
        self.cache = cache
        self.regenerate = regenerate
        self.years = years or [2000]  # Default to one year if not specified
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_data(self, shape):
        if self.distribution == "normal":
            data = np.random.normal(loc=self.mean, scale=self.std, size=shape)
        elif self.distribution == "uniform":
            data = np.random.uniform(low=self.mean - self.std, high=self.mean + self.std, size=shape)
        elif self.distribution == "lognormal":
            data = np.random.lognormal(mean=self.mean, sigma=self.std, size=shape)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        if self.spatial != self.target_spatial:
            zoom_factors = (1, self.target_spatial[0] / self.spatial[0], self.target_spatial[1] / self.spatial[1], 1)
            data = zoom(data, zoom_factors, order=1)
        
        return data

    def save_hdf5(self, data, file_path):
        with h5py.File(file_path, "w") as f:
            f.create_dataset("data", data=data, compression="gzip")

    def save_zarr(self, data, file_path):
        zarr.save(file_path, data)

    def generate(self):
        for year in self.years:
            file_name = f"test_data_{year}.{self.format}"
            file_path = self.output_dir / file_name
            
            if self.cache and file_path.exists() and not self.regenerate:
                print(f"Using cached file: {file_path}")
                continue
            
            shape = (self.temporal,) + self.spatial + (self.channels,)
            data = self.generate_data(shape)
            
            if self.format == "hdf5":
                self.save_hdf5(data, file_path)
            elif self.format == "zarr":
                self.save_zarr(data, file_path)
            
            print(f"Generated test data: {file_path}")

class InMemoryDataset(Dataset):
    def __init__(self, mean_file, std_file, spatial=(100, 100), target_spatial=None, temporal=10, channels=5, distribution="normal"):
        self.mean = np.load(mean_file)
        self.std = np.load(std_file)
        self.spatial = spatial
        self.target_spatial = target_spatial if target_spatial else spatial
        self.temporal = temporal
        self.channels = channels
        self.distribution = distribution
        
        self.data = self.generate_data()
    
    def generate_data(self):
        shape = (self.temporal,) + self.spatial + (self.channels,)
        if self.distribution == "normal":
            data = np.random.normal(loc=self.mean, scale=self.std, size=shape)
        elif self.distribution == "uniform":
            data = np.random.uniform(low=self.mean - self.std, high=self.mean + self.std, size=shape)
        elif self.distribution == "lognormal":
            data = np.random.lognormal(mean=self.mean, sigma=self.std, size=shape)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        if self.spatial != self.target_spatial:
            zoom_factors = (1, self.target_spatial[0] / self.spatial[0], self.target_spatial[1] / self.spatial[1], 1)
            data = zoom(data, zoom_factors, order=1)
        
        return torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return self.temporal
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    generator = TestDataGenerator(
        mean_file="mean.npy", std_file="std.npy", format="zarr", 
        spatial=(328, 720), target_spatial=(240, 480), temporal=10, channels=5, years=[2000, 2001]
    )
    generator.generate()
    
    dataset = InMemoryDataset(
        mean_file="mean.npy", std_file="std.npy", spatial=(328, 720), target_spatial=(240, 480), temporal=10, channels=5
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch in dataloader:
        print(batch.shape)
