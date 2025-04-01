import glob
import numpy as np
import os
import random 


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

    
class TimeSeriesDataset(Dataset):
    def __init__(self, data, gt, gesture_list, target_cycle_len, downsample_rate=1, is_train=True, mean=None, std=None):
        """
        Args:
            data: nested dictionary in the form of {sample_code: {gesture: [cycle1, cycle2, ...]}}
            gt: ground truth dictionary of pairs [sample_code, label]
            downsample_rate: rate for reducing data frequency 
            gesture_list: a list of available gesture names
            target_cycle_len: the target length to pad each cycle before downsampling 
            downsample_rate: reduce data frequency by a factor of downsample_rate
            is_train: True if the dataset is for training, False otherwise.
            mean: an array of mean values with size [n_gestures, n_channels]. Compute it from data if it is None
            std: an array of standard deviation values with size [n_gestures, n_channels]. Compute it from data if it is None
        """
        self.data = data
        self.gt = gt
        self.gesture_list = gesture_list
        self.target_cycle_len = target_cycle_len//downsample_rate
        self.downsample_rate = downsample_rate
        self.is_train = is_train
        self.sc_list = list(gt.keys())
        self.label_list = np.unique(list(gt.values()))
        
        if mean is None or std is None:
            self.mean = []
            self.std = []
            # Compute gesture-wise and channel-wise mean and std
            for gesture in gesture_list:
                self.mean.append(np.mean(np.vstack([np.vstack(v[gesture]) for v in data.values()]),axis=0))
                self.std.append(np.std(np.vstack([np.vstack(v[gesture]) for v in data.values()]),axis=0))
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)

        else:
            self.mean = mean
            self.std = std
        # Standardize kinematic data, gesture-wise and channel-wise.
        for sc in self.data:
            for ges_idx, gesture in enumerate(gesture_list):
                for cycle_idx in range(len(self.data[sc][gesture])):
                    self.data[sc][gesture][cycle_idx] = (self.data[sc][gesture][cycle_idx]-self.mean[ges_idx])/self.std[ges_idx]
    def __len__(self):
        if self.is_train:
            return 10000 # Set to a large number, and use max_step to control the number of dataloader loops
        else:
            return len(self.sc_list)

    def __getitem__(self, idx):
        if self.is_train:
            # During training, randomly choose a class then pick a random sample in the class
            sc, label = self.uniform_sample()
        else:
            # During testing, explicitly pick sample according to sample codes.
            sc = self.sc_list[idx]
            label = self.gt[sc]
        max_len = 0
        padded_data = np.zeros((len(self.gesture_list), self.target_cycle_len, 8), dtype=float)
        # Create batch of data by randomly choose one cycle of data from each gesture each time.
        random.seed() # Select a random seed each time so that different workers output different samples.
        for i, gesture in enumerate(self.gesture_list):        
            cycle_idx = random.randint(0, len(self.data[sc][gesture])-1)
            cycle = np.copy(self.data[sc][gesture][cycle_idx])
            # reduce frequency of each cycle
            cycle = cycle[::self.downsample_rate]
            if self.is_train:
                # Add random noise
                cycle += np.random.normal(0, 0.1, cycle.shape) 
            padded_data[i, :len(cycle)] = cycle
            max_len = max(max_len, len(cycle))

        padded_data = np.moveaxis(padded_data, 2, 1) # [n_gesture, time, channel] -> [n_gesture, channel, time]
        if self.is_train:
            # randomly shift data on time dimension
            shift_size = random.randint(0, self.target_cycle_len-max_len)
            padded_data = np.roll(padded_data, shift=shift_size, axis=2)
        return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    def uniform_sample(self):
        # Random choose a label, then random choose a sc from the label class
        random.seed() # Select a random seed each time so that different workers output different samples.
        label = random.choice(self.label_list)
        candidates = [key for key in self.gt if self.gt[key] == label]
        sc = random.choice(candidates)
        return sc, label
    
'''
# Dataset that sample a batch of data from one subject each time. 
# the batch data is padded to the max length of the cycle in this batch
# Does not work well for training.
class TimeSeriesDataset(Dataset):
    def __init__(self, data, gt, gesture_list, target_cycle_len, batch_size, downsample_rate=1, mean=None, std=None):
        """
        Args:
            data: nested dictionary in the form of {sample_code: {gesture: [cycle1, cycle2, ...]}}
            gt: ground truth dictionary of pairs [sample_code, label]
            downsample_rate: rate for reducing data frequency 
            gesture_list: a list of available gesture names
            target_cycle_len: the target length to pad each cycle before downsampling 
            batch_size: number of samples to generate in each batch. The dataset class output batch data directly
            downsample_rate: reduce data frequency by a factor of downsample_rate
            mean: an array of mean values with size [n_gestures, n_channels]. Compute it from data if it is None
            std: an array of standard deviation values with size [n_gestures, n_channels]. Compute it from data if it is None
        """
        self.data = data
        self.gt = gt
        self.gesture_list = gesture_list
        self.target_cycle_len = target_cycle_len
        self.batch_size = batch_size
        self.downsample_rate = downsample_rate
        self.sc_list = list(self.gt.keys())
        if mean is None or std is None:
            self.mean = []
            self.std = []
            # Compute gesture-wise and channel-wise mean and std
            for gesture in gesture_list:
                self.mean.append(np.mean(np.vstack([np.vstack(v[gesture]) for v in data.values()]),axis=0))
                self.std.append(np.std(np.vstack([np.vstack(v[gesture]) for v in data.values()]),axis=0))
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)

        else:
            self.mean = mean
            self.std = std
        # Standardize kinematic data, gesture-wise and channel-wise.
        for sc in self.data:
            for ges_idx, gesture in enumerate(gesture_list):
                for cycle_idx in range(len(self.data[sc][gesture])):
                    self.data[sc][gesture][cycle_idx] = (self.data[sc][gesture][cycle_idx]-self.mean[ges_idx])/self.std[ges_idx]
    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        sc = self.sc_list[idx]
        label = [self.gt[sc]] * self.batch_size
        max_len = 0 # Cycle data from the same subject id always have simliar durations
        cycle_data = []
        # Create batch of data by randomly choose one cycle of data from each gesture each time.
        batch_data = []
        for _ in range(self.batch_size):
            for gesture in self.gesture_list:            
                cycle_idx = random.randint(0, len(self.data[sc][gesture])-1)
                cycle = self.data[sc][gesture][cycle_idx]
                # reduce frequency of each cycle
                cycle = cycle[::self.downsample_rate]
                cycle_data.append(cycle)
                max_len = max(max_len, len(cycle))   
            batch_data.append(cycle_data)
            
        # Pad each cycle to the same length
        padded_data = np.zeros((self.batch_size, len(self.gesture_list), max_len, 8), dtype=float)
        for batch_i, cycle_data in enumerate(batch_data):
            for gesture_i in range(len(self.gesture_list)):
                cycle = batch_data[batch_i][gesture_i]
                padded_data[batch_i, gesture_i, :len(cycle)] = cycle
        padded_data = np.moveaxis(padded_data, 3, 2) # [batch, n_gesture, time, channel] -> [batch, n_gesture, channel, time]
        return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
'''