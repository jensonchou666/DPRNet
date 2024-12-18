from torch.utils.data import Dataset
from typing import List

class ZipDataset(Dataset):
    def __init__(self, datasets: List[Dataset], transforms=None, assert_equal_length=False):
        self.datasets = datasets
        self.transforms = transforms
        
        if assert_equal_length:
            for i in range(1, len(datasets)):
                assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'
    
    def __len__(self):
        return max(len(d) for d in self.datasets)
    
    def __getitem__(self, idx):
        x = tuple(d[idx % len(d)] for d in self.datasets)
        if self.transforms:
            x = self.transforms(*x)
        return x


class DictDataset(Dataset):
    def __init__(self, dataset, keys):
        self.dataset = dataset
        self.keys = keys
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x = self.dataset.__getitem__(idx)
        return {k:y for k, y in zip(self.keys, x)}