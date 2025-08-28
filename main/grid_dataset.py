import torch
from torch.utils.data import Dataset
import numpy as np

class GridSequenceDataset(Dataset):
    def __init__(self, sequence_paths, transform=None):
        """
        sequence_paths: lista de tuplas con 4 paths (3 inputs, 1 target)
        transform: transformación opcional a aplicar (ej: normalización, resize)
        """
        self.sequence_paths = sequence_paths
        self.transform = transform

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        input_paths = self.sequence_paths[idx][:3]
        target_path = self.sequence_paths[idx][3]

        inputs = [np.load(p) for p in input_paths]
        target = np.load(target_path)

        # Asegurarse que tengan forma [1, H, W] si son 2D
        inputs = [inp[None, :, :] if inp.ndim == 2 else inp for inp in inputs]
        target = target[None, :, :] if target.ndim == 2 else target

        input_tensor = torch.from_numpy(np.concatenate(inputs, axis=0)).float()  # Shape: [3, H, W]
        target_tensor = torch.from_numpy(target).float()                         # Shape: [1, H, W]

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor, target_tensor
