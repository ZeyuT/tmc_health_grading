# Zeyu Tang
# 07/2025

# run.py
# Purpose: Train and test CNN-LSTM models using kinematic motion data.
# Usage: python run.py 
# configs are stored in ../configs.yaml

import numpy as np
import os
import sys
import random 

import pandas as pd
import imageio
import csv
import glob
import yaml
import pickle
import math
import collections
from itertools import cycle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import optim

from torchvision import transforms
from torchinfo import summary
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score

from data_loader import TimeSeriesTrainingSet, TimeSeriesTestingSet, TimeSeriesReferenceSet
from model import *
from train import train
from test import test
import argparse
import gc 
import copy

def main():
    
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel_idx', '--c', type=int, default=None, help="Index of the feature channel used")
    args = parser.parse_args()

    channel_idx = args.channel_idx
    if channel_idx == -1:
        input_channels = 2
    elif channel_idx is None:
        input_channels = 8
    else:
        input_channels = 1
    config_path = '../configs.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    model_configs = configs['kinematic_model_configs']
    gt_type = model_configs['gt_type']
    cv_configs = configs['cv_configs']
    results_path = os.path.join(configs['results_path'], 'kinematic_model')

    os.makedirs(results_path, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'start training kinematic models on {torch.cuda.get_device_name(torch.cuda.current_device())}')
    
    # Load ground truth labels and data
    with open(configs['gt_path'], 'rb') as file:
        grades = pickle.load(file)

    # Manually select which type of labels to use
    # Options: group, kinematic, bony, ligament, old
    # (old: old 5-class scheme, group: proposed 3-class scheme) 
    # (kinematic, bony, ligament: binary labels for corresponding aspects)
    gt_labels = grades[gt_type]
    all_hid = np.array(list(gt_labels.keys()))
    
    with open(os.path.join(configs['processed_data_path'], 'kinematic_data.pkl'), 'rb') as file:
        data = pickle.load(file)
    valid_hid = np.array(list(data.keys()))
        
    kf = KFold(n_splits=cv_configs['num_splits'], 
               shuffle=True, 
               random_state=cv_configs['random_seed'])

    # Open a CSV file to write the results
    result_file = open(os.path.join(results_path, f"result_{gt_type}_ch_{channel_idx if channel_idx is not None else 'all'}.csv"), mode='w', newline='')
    writer = csv.writer(result_file)
    writer.writerow(['Fold', 'Sample Code', 'Prediction', 'Ground Truth']+[f'Closest_{i+1} hid(gt) distance' for i in range(model_configs['top_k'])])

    # Main loop for K-fold Cross Validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_hid)):
        print(f"Fold {fold + 1}/{cv_configs['num_splits']}")
    
        train_gt = {key: gt_labels[key] for key in all_hid[train_idx] if key in valid_hid}
        train_data = {key: data[key] for key in all_hid[train_idx] if key in valid_hid}
        
        test_gt = {key: gt_labels[key] for key in all_hid[test_idx] if key in valid_hid}
        test_data = {key: data[key] for key in all_hid[test_idx] if key in valid_hid}
        
        train_set = TimeSeriesTrainingSet(
            data=copy.deepcopy(train_data),
            gt=copy.deepcopy(train_gt),
            gesture_list=configs['gesture_list'], 
            target_cycle_len=configs['target_cycle_len'], 
            downsample_rate=configs['downsample_rate'],
            channel_idx=channel_idx,
            )
        
        train_loader = DataLoader(train_set, 
                                  batch_size=model_configs['batch_size'],
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=False)
        # Model setup for the current fold
        kinematic_model = KinematicModel(n_gestures=5, input_channels=input_channels, 
                                         feat_channel=8)
        kinematic_model = kinematic_model.apply(initialize_weights)
        kinematic_model = kinematic_model.to(device)
        
        # Training phase
        model_path = os.path.join(results_path, f'model_{gt_type}_fold_{fold+1}.pth')
        if os.path.exists(model_path):
            os.remove(model_path)
        kinematic_model = train(train_loader, kinematic_model, model_configs, model_path, device)
        
        # Load weights from the saved best model
        kinematic_model.load_state_dict(torch.load(model_path))

        # Evaluation phase
        # Define testing set for the current fold.
        test_set = TimeSeriesTestingSet(
                        data=copy.deepcopy(test_data),
                        gt=copy.deepcopy(test_gt),
                        gesture_list=configs['gesture_list'], 
                        target_cycle_len=configs['target_cycle_len'], 
                        downsample_rate=configs['downsample_rate'],
                        channel_idx = channel_idx,
                        )  
        reference_set = TimeSeriesReferenceSet(
                        data=copy.deepcopy(train_data),
                        gt=copy.deepcopy(train_gt),
                        gesture_list=configs['gesture_list'], 
                        target_cycle_len=configs['target_cycle_len'], 
                        downsample_rate=configs['downsample_rate'],
                        channel_idx = channel_idx,
                        )
        test(kinematic_model, reference_set, test_set, writer, fold, device, model_configs['top_k'])
        
        del train_set, test_set, reference_set, train_loader, kinematic_model
        torch.cuda.empty_cache()
        gc.collect()
        
    result_file.close() 

if __name__ == "__main__":  
    main()