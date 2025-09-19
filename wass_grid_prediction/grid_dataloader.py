import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from typing import Tuple

class GridDataset(Dataset):
    def __init__(self, nc_path: str, split: str = 'train', split_ratio: float = 0.8, transform=None):
        """
        Args:
            nc_path (str): Path to the NetCDF file.
            split (str): 'train' or 'valid'.
            split_ratio (float): Fraction of data to use for training.
            transform: Optional transform to be applied on a sample.
        """
        self.ds = xr.open_dataset(nc_path, engine='netcdf4', decode_cf=False)
        self.data = self.ds['Z']  # shape: (time, N, M)
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.scale_factor = self.data.attrs.get('scale_factor', 1.0)
        self.add_offset = self.data.attrs.get('add_offset', 0.0)
        self._prepare_indices()

    def _prepare_indices(self):
        total_frames = self.data.shape[0]
        # We need at least 3 frames for input, so last usable index is total_frames-1
        # For input: t-3, t-2, t-1; target: t, so usable indices start at 3
        usable_indices = np.arange(3, total_frames)
        split_point = int(len(usable_indices) * self.split_ratio)
        if self.split == 'train':
            self.indices = usable_indices[:split_point]
        else:
            self.indices = usable_indices[split_point:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[idx]
        # Input: frames t-3, t-2, t-1; Target: frame t
        input_frames = self.data.isel(time=slice(t-3, t)).values  # shape: (3, N, M)
        target_frame = self.data.isel(time=t).values  # shape: (N, M)
        # Apply scale/offset
        input_frames = (input_frames * self.scale_factor) + self.add_offset
        target_frame = (target_frame * self.scale_factor) + self.add_offset
        # Masking (optional, if mask_Z or _FillValue exists)
        fill_value = self.data.attrs.get('_FillValue', None)
        if fill_value is not None:
            input_frames = np.ma.masked_equal(input_frames, fill_value).filled(np.nan)
            target_frame = np.ma.masked_equal(target_frame, fill_value).filled(np.nan)
        # Fill NaNs with channel mean for input, and mean for target
        if np.isnan(input_frames).any():
            # Compute mean for each channel (ignoring NaNs)
            for c in range(input_frames.shape[0]):
                channel = input_frames[c]
                mean_val = np.nanmean(channel)
                channel[np.isnan(channel)] = mean_val
                input_frames[c] = channel
        if np.isnan(target_frame).any():
            mean_val = np.nanmean(target_frame)
            target_frame[np.isnan(target_frame)] = mean_val
        # Convert to torch.Tensor
        input_tensor = torch.from_numpy(input_frames).float()
        target_tensor = torch.from_numpy(target_frame).float()
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
        return input_tensor, target_tensor

def get_dataloaders(nc_path: str, batch_size: int = 8, split_ratio: float = 0.8, num_workers: int = 0):
    train_dataset = GridDataset(nc_path, split='train', split_ratio=split_ratio)
    valid_dataset = GridDataset(nc_path, split='valid', split_ratio=split_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader

if __name__ == "__main__":
    nc_file = "Surfaces_20150305_103500.nc"  # Change to your file path
    train_loader, valid_loader = get_dataloaders(nc_file, batch_size=4)
    # Example: iterate through one batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: inputs shape {inputs.shape}, targets shape {targets.shape}")
        break
