from torch.utils.data import Dataset
import random
import torch
import re
import os
import gc


class R1Dataset(Dataset):
    def __init__(self, data_dir, nsamples, shuffle=False):
        self.data_dir = data_dir 
        self.file_list = self._load_file_list(nsamples)  
        self.shuffle = shuffle

    def _load_file_list(self, nsamples: int):
        file_list = [
            f for f in os.listdir(self.data_dir)
            if f.endswith('.pt') and re.match(r'sample_(\d{1,2})_', f) and 0 <= int(re.match(r'sample_(\d{1,2})_', f).group(1)) <= nsamples - 1
        ]
        file_list.sort()
        return file_list

    def _shuffle_data(self, sample):
        """Shuffle num_b and tokens dimensions."""
        perm1 = torch.randperm(self.current_data.shape[0])
        sample = sample[perm1, :]
        return sample

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
    
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample = torch.load(file_path, map_location='cpu', weights_only=True)  
        if self.shuffle:
            sample = self._shuffle_data(sample)
        return sample


class R2Dataset(Dataset):
    def __init__(self, data_path, nsamples, dev):
        self.data = torch.load(data_path, map_location=dev)[:nsamples, :, :]
        self.data = self.data[torch.randperm(self.data.size(0)), :, :]
        self.data = self.data[:, torch.randperm(self.data.size(1)), :]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx] 
