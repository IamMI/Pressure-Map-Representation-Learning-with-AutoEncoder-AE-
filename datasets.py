"""
Make datasets
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np

class datasets(Dataset):
    def __init__(self, opt, phase):
        self.phase = phase
        self.images_fn = np.load(opt['tv_fn'], allow_pickle=True).item()
        self.images_fn = self.images_fn[self.phase]
        self.press = np.load(os.path.join('./mini_data', 'pressure.npz'), allow_pickle=True)['pressure']
        self.max_len = 40
        
    def __len__(self):
        return len(self.images_fn)

    def __getitem__(self, index):
        base, num = os.path.split(self.images_fn[index])
        num = int(num)

        press = self.press[num*self.max_len: self.max_len*(num+1)]/255
        press = torch.from_numpy(press).float()
        
        result = {
            'press': press,
        }
        return result
