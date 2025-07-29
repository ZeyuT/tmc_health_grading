import glob
import numpy as np
import os
import random 


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from scipy.interpolate import interp1d

    
class TimeSeriesTrainingSet(Dataset):
    def __init__(self, data, gt, gesture_list, target_cycle_len, 
                 downsample_rate=1, 
                 std=None, mean=None,
                 channel_idx=None):
        """
        Args:
            data: nested dictionary in the form of {sample_code: {gesture: [cycle1, cycle2, ...]}}
            gt: ground truth dictionary of pairs [sample_code, label]
            downsample_rate: rate for reducing data frequency 
            gesture_list: a list of available gesture names
            target_cycle_len: the target length to pad each cycle before downsampling 
            downsample_rate: reduce data frequency by a factor of downsample_rate
            std: an array of standard deviation values with size [n_gestures, n_channels]. Compute it from data if it is None
            channel_idx: idx of feature channel to use. Default: None (use all channels)
        """
        self.data = data
        self.gt = gt
        self.gesture_list = gesture_list
        self.target_cycle_len = target_cycle_len
        self.downsample_rate = downsample_rate
        self.channel_idx = channel_idx
        self.hid_list = list(gt.keys())
        self.label_list = np.unique(list(gt.values()))
        '''
        # Compute gesture-wise and channel-wise std, and preserve relative magnitudes between samples
        self.std = []
        self.mean = []
        for gesture in gesture_list:
            self.std.append(np.std(np.vstack([np.vstack(subject[gesture]) for subject in data.values() if gesture in subject]),axis=0))
            self.mean.append(np.mean(np.vstack([np.vstack(subject[gesture]) for subject in data.values() if gesture in subject]),axis=0))
        self.std = np.array(self.std)
        self.mean = np.array(self.mean)
        '''
        
        # Standardize kinematic data.
        for hid in self.data:
            for gesture in self.data[hid]:
                for cycle_idx in range(len(self.data[hid][gesture])):
                    cur_min = np.min(self.data[hid][gesture][cycle_idx], axis=0) # zero-min each cycle
                    self.data[hid][gesture][cycle_idx] = (self.data[hid][gesture][cycle_idx]-cur_min)/180
    def __len__(self):
        return 100000 # Set to a large number, and use max_step to control the number of dataloader loops

    def __getitem__(self, idx):
        # During training, return a pair of samples
        # Random sample labels and hid 
        hid1, label1 = self.uniform_sample_label()
        hid2, label2 = self.uniform_sample_label()
        sample1 = self.build_sample(hid1)
        sample2 = self.build_sample(hid2)
        if label1 == label2:
            # Similar pair: 0 
            pair_label = torch.tensor(0)
        else:
            # Dissimilar pair: 1 
            pair_label = torch.tensor(1)
        return sample1, sample2, pair_label
    
    def build_sample(self, hid):
        # Create a sample by randomly choose one cycle from each gesture of [hid] subject
        padded_data = np.zeros((len(self.gesture_list), self.target_cycle_len//self.downsample_rate, 8), dtype=float)
        
        random.seed() # Select a random seed each time so that different workers output different samples.
        max_len = 0
        for i, gesture in enumerate(self.gesture_list):       
            if gesture not in self.data[hid]:
                continue
            cycle_idx = random.randint(0, len(self.data[hid][gesture])-1)
            cycle = np.copy(self.data[hid][gesture][cycle_idx])
            '''
            # Upsample to target_len
            cycle = self.upsample(cycle, self.target_cycle_len)
            '''
            # Then downsample each cycle by 10x
            cycle = cycle[::self.downsample_rate]
            # Add random noise
            # cycle += np.random.normal(0, 0.1, cycle.shape) 
            padded_data[i, :len(cycle)] = cycle
            max_len = max(max_len, len(cycle))
            
        padded_data = np.moveaxis(padded_data, 2, 1) # [n_gesture, time, channel] -> [n_gesture, channel, time]
        
        # randomly shift data on time dimension
        shift_size = random.randint(0, self.target_cycle_len-max_len)
        padded_data = np.roll(padded_data, shift=shift_size, axis=2)
        '''
        # Select channels to use
        if self.channel_idx == -1:
            padded_data = padded_data[:, [4,6], :]
        elif self.channel_idx is not None:
            padded_data = padded_data[:, self.channel_idx:self.channel_idx+1, :]
        '''
        return torch.tensor(padded_data, dtype=torch.float32)
    
    def uniform_sample_label(self):
        # Random choose a label, then random choose a hid from the label class
        random.seed() # Select a random seed each time so that different workers output different samples.
        label = random.choice(self.label_list)
        candidates = [key for key in self.gt if self.gt[key] == label]
        hid = random.choice(candidates)
        return hid, label
    
    def upsample(self, signal, target_len):
        # Upsample a 1D signal to target_len using linear interpolation.
        ori_len, channels = signal.shape
        if ori_len== target_len:
            return signal
        x_old = np.linspace(0, 1,ori_len)
        x_new = np.linspace(0, 1, target_len)
        upsampled = np.zeros((target_len, channels), dtype=signal.dtype)

        for c_idx in range(channels):
            f = interp1d(x_old, signal[:, c_idx], kind='linear', fill_value="extrapolate")
            upsampled[:, c_idx] = f(x_new)

        return upsampled

class TimeSeriesTestingSet(Dataset):
    def __init__(self, data, gt, gesture_list, target_cycle_len, 
                 downsample_rate=1,
                 std=None, mean=None,
                 channel_idx=None):
        """
        Args:
            data: nested dictionary in the form of {sample_code: {gesture: [cycle1, cycle2, ...]}}
            gt: ground truth dictionary of pairs [sample_code, label]
            downsample_rate: rate for reducing data frequency 
            gesture_list: a list of available gesture names
            target_cycle_len: the target length to pad each cycle before downsampling 
            downsample_rate: reduce data frequency by a factor of downsample_rate
            is_train: True if the dataset is for training, False otherwise.
            std: an array of standard deviation values with size [n_gestures, n_channels]. Compute it from data if it is None
            channel_idx: idx of feature channel to use. Default: None (use all channels)
        """
        self.data = data
        self.gt = gt
        self.gesture_list = gesture_list
        self.target_cycle_len = target_cycle_len
        self.downsample_rate = downsample_rate
        self.channel_idx = channel_idx
        self.hid_list = list(gt.keys())
        self.label_list = np.unique(list(gt.values()))
        
        '''
        self.std = std
        self.mean = mean
        '''
        # Standardize kinematic data.
        for hid in self.data:
            for gesture in self.data[hid]:
                for cycle_idx in range(len(self.data[hid][gesture])):
                    cur_min = np.min(self.data[hid][gesture][cycle_idx], axis=0) # zero-min each cycle
                    self.data[hid][gesture][cycle_idx] = (self.data[hid][gesture][cycle_idx]-cur_min)/180
    def __len__(self):
        return len(self.hid_list)

    def __getitem__(self, idx):
        # During testing, explicitly pick sample according to sample codes.
        hid = self.hid_list[idx]
        label = self.gt[hid]
        padded_data = np.zeros((len(self.gesture_list), self.target_cycle_len//self.downsample_rate, 8), dtype=float)
        # Create batch of data by randomly choose one cycle of data from each gesture each time.
        random.seed() # Select a random seed each time so that different workers output different samples.
        max_len = 0
        for i, gesture in enumerate(self.gesture_list):   
            if gesture not in self.data[hid]:
                continue
            cycle_idx = random.randint(0, len(self.data[hid][gesture])-1)
            cycle = np.copy(self.data[hid][gesture][cycle_idx])
            '''
            # Upsample each cycle to target_len
            cycle = self.upsample(cycle, self.target_cycle_len)
            '''
            # Then downsample each cycle
            cycle = cycle[::self.downsample_rate]
            padded_data[i, :len(cycle)] = cycle
            max_len = max(max_len, len(cycle))

        padded_data = np.moveaxis(padded_data, 2, 1) # [n_gesture, time, channel] -> [n_gesture, channel, time]
        
        '''
        # Select channels to use
        if self.channel_idx == -1:
            padded_data = padded_data[:, [4,6], :]
        elif self.channel_idx is not None:
            padded_data = padded_data[:, self.channel_idx:self.channel_idx+1, :]
        '''
        return torch.tensor(padded_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def upsample(self, signal, target_len):
        # Upsample a 1D signal to target_len using linear interpolation.
        ori_len, channels = signal.shape
        if ori_len== target_len:
            return signal
        x_old = np.linspace(0, 1,ori_len)
        x_new = np.linspace(0, 1, target_len)
        upsampled = np.zeros((target_len, channels), dtype=signal.dtype)

        for c_idx in range(channels):
            f = interp1d(x_old, signal[:, c_idx], kind='linear', fill_value="extrapolate")
            upsampled[:, c_idx] = f(x_new)

        return upsampled
    
class TimeSeriesReferenceSet:
    def __init__(self, data, gt, gesture_list, target_cycle_len, 
                 downsample_rate=1, 
                 std=None, mean=None,
                 channel_idx=None):
        """
        Args:
            data: nested dictionary in the form of {sample_code: {gesture: [cycle1, cycle2, ...]}}
            gt: ground truth dictionary of pairs [sample_code, label]
            downsample_rate: rate for reducing data frequency 
            gesture_list: a list of available gesture names
            target_cycle_len: the target length to pad each cycle before downsampling 
            downsample_rate: reduce data frequency by a factor of downsample_rate
            std: an array of standard deviation values with size [n_gestures, n_channels]. Compute it from data if it is None
            channel_idx: idx of feature channel to use. Default: None (use all channels)
        """
        self.data = data
        self.gt = gt
        self.gesture_list = gesture_list
        self.target_cycle_len = target_cycle_len
        self.downsample_rate = downsample_rate
        self.channel_idx = channel_idx
        self.hid_list = list(gt.keys())
        self.label_list = np.unique(list(gt.values()))
        
        # Standardize kinematic data.
        for hid in self.data:
            for gesture in self.data[hid]:
                for cycle_idx in range(len(self.data[hid][gesture])):
                    cur_min = np.min(self.data[hid][gesture][cycle_idx], axis=0) # zero-center each cycle
                    self.data[hid][gesture][cycle_idx] = (self.data[hid][gesture][cycle_idx]-cur_min)/180

    def get_data(self):
        samples = []
        labels = []
        hid_list = []
        for hid in self.gt:
            samples.append(self.build_sample(hid))
            labels.append(torch.tensor(self.gt[hid], dtype=torch.long))
            hid_list.append(hid)

        return torch.stack(samples), np.array(labels), hid_list

    def build_sample(self, hid):
        # Create a sample by randomly choose one cycle from each gesture of [hid] subject
        padded_data = np.zeros((len(self.gesture_list), self.target_cycle_len//self.downsample_rate, 8), dtype=float)
        
        random.seed() # Select a random seed each time so that different workers output different samples.
        max_len = 0
        for i, gesture in enumerate(self.gesture_list):       
            if gesture not in self.data[hid]:
                continue
            cycle_idx = random.randint(0, len(self.data[hid][gesture])-1)
            cycle = np.copy(self.data[hid][gesture][cycle_idx])
            '''
            # Upsample each cycle to target_len
            cycle = self.upsample(cycle, self.target_cycle_len)
            '''
            # Then downsample each cycle
            cycle = cycle[::self.downsample_rate]
            padded_data[i, :len(cycle)] = cycle
            max_len = max(max_len, len(cycle))
            
        padded_data = np.moveaxis(padded_data, 2, 1) # [n_gesture, time, channel] -> [n_gesture, channel, time]
        
        '''
        # Select channels to use
        if self.channel_idx == -1:
            padded_data = padded_data[:, [4,6], :]
        elif self.channel_idx is not None:
            padded_data = padded_data[:, self.channel_idx:self.channel_idx+1, :]
        '''
        return torch.tensor(padded_data, dtype=torch.float32)

    def upsample(self, signal, target_len):
        # Upsample a 1D signal to target_len using linear interpolation.
        ori_len, channels = signal.shape
        if ori_len== target_len:
            return signal
        x_old = np.linspace(0, 1,ori_len)
        x_new = np.linspace(0, 1, target_len)
        upsampled = np.zeros((target_len, channels), dtype=signal.dtype)

        for c_idx in range(channels):
            f = interp1d(x_old, signal[:, c_idx], kind='linear', fill_value="extrapolate")
            upsampled[:, c_idx] = f(x_new)

        return upsampled