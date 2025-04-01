import glob
import numpy as np
import os
import random

import imageio
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, gt, data_path):
        """
        Args:
            gt: ground truth dictionary of pairs [sample_code, label]
            root_path: path to the data folder
        """
        self.data_path = data_path

        self.gt = gt
        self.sc_list = list(gt.keys())
        self.label_list = np.unique(list(gt.values()))
        
        # Load all data into memory
        self.data = {}
        for sc in self.sc_list:
            with open(os.path.join(self.data_path, f'{sc}.pkl'), 'rb') as file:
                self.data[sc] = pickle.load(file)
            
    def __len__(self):
        return len(self.sc_list)

    def __getitem__(self, idx):
        sc, label = self.uniform_sample()
        data = np.copy(self.data[sc])
        # Normalization
        data = (data-data.min())/(data.max() - data.min())
        data = data[::4, ::4, ::4] # resize to 128x128x128
        data += np.random.normal(0, 0.1, data.shape) 
        data = np.expand_dims(data, 0) # Add the channel dimension
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def uniform_sample(self):
        # Random choose a label, then random choose a sc from the label class
        random.seed() # Select a random seed each time so that different workers output different samples.
        label = random.choice(self.label_list)
        candidates = [key for key in self.gt if self.gt[key] == label]
        sc = random.choice(candidates)
        return sc, label