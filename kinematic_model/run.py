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

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score

from data_loader import TimeSeriesDataset
from model import *


def main():
    grade_idx = int(sys.argv[1]) # 1: old grading scheme. 2: new grading scheme
    
    # Configurations
    config_path = '/project/ahoover/mhealth/zeyut/tmc/TMC AI Files/configs.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    model_configs = configs['kinematic_model_configs']
    cv_configs = configs['cv_configs']
    results_path = os.path.join(configs['results_path'], 'kinematic_model')

    os.makedirs(results_path, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'start training kinematic models on {torch.cuda.get_device_name(torch.cuda.current_device())}')
    # Load ground truth labels and data
    with open(configs['gt_path'], 'rb') as file:
        grades = pickle.load(file)
    if grade_idx == 1:
        gt_labels = grades['old']
    else:
        gt_labels = {}
        for sc in grades['new'].keys():
            if grades['new'][sc] == 1:
                gt_labels[sc] = 1
            else:
                gt_labels[sc] = 0
        
    with open(os.path.join(configs['processed_data_path'], 'kinematic_data.pkl'), 'rb') as file:
        data = pickle.load(file)
    all_sc = np.array(list(gt_labels.keys()))
    valid_sc = np.array(list(data.keys()))

    num_classes = len(np.unique(list(gt_labels.values())))
    kf = KFold(n_splits=cv_configs['num_splits'], 
               shuffle=True, 
               random_state=cv_configs['random_seed'])
 
    # Open a CSV file to write the results
    result_file = open(os.path.join(results_path, f'result_{grade_idx}.csv'), mode='w', newline='')
    writer = csv.writer(result_file)
    writer.writerow(['Fold', 'Sample Code', 'Prediction', 'Ground Truth']+[f'P{class_}' for class_ in range(num_classes)])

    # Main loop for K-fold Cross Validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_sc)):
        print(f"Fold {fold + 1}/{cv_configs['num_splits']}")

        # Define datasets for the current fold
        train_gt = {key: gt_labels[key] for key in all_sc[train_idx] if key in valid_sc}
        train_data = {key: data[key] for key in all_sc[train_idx] if key in valid_sc}
        train_set = TimeSeriesDataset(
            data=train_data,
            gt=train_gt,
            gesture_list=configs['gesture_list'], 
            target_cycle_len=configs['target_cycle_len'], 
            mean=None, std=None,
            downsample_rate=configs['downsample_rate']
            )
        
        
        # Create data loaders for the current fold
        train_loader = DataLoader(train_set, 
                                  batch_size=model_configs['batch_size'],
                                  shuffle=True, 
                                  num_workers=4, 
                                  pin_memory=False)
        # Model setup for the current fold
        kinematic_model = KinematicModel(n_gestures=5, input_channels=8, 
                                         feat_channel=8, 
                                         num_classes=num_classes)
        kinematic_model = kinematic_model.apply(initialize_weights)
        kinematic_model = kinematic_model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(kinematic_model.parameters(), 
                         lr=model_configs['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=model_configs['patience'], 
                                                         verbose=True)
        # Training phase
        kinematic_model.train()
        best_loss = 10000
        for epoch in range(model_configs['num_epochs']):
            total_loss = 0
            steps = 0
            corrects = 0
            total_samples = 0
            for samples, labels in train_loader:
                samples = samples.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = kinematic_model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                preds = torch.argmax(outputs, 1)                
                corrects += (preds == labels).sum().item()
                total_samples += labels.size(0)
                steps += 1
                if steps > model_configs['max_step']:
                    break
            train_loss = total_loss / steps
            train_acc = corrects / total_samples
            scheduler.step(train_loss)
            print(f"Epoch {epoch + 1}, Accuracy: {train_acc*100:.2f}%, Loss: {train_loss:.4f}")
            sys.stdout.flush()
            
            if best_loss > train_loss:               
                # Save the best model for each fold
                torch.save(kinematic_model.state_dict(), os.path.join(results_path, f'model_{grade_idx}_fold_{fold+1}.pth'))    
                best_loss = train_loss
        # Load weights from the saved best model
        kinematic_model.load_state_dict(torch.load(os.path.join(results_path, f'model_{grade_idx}_fold_{fold+1}.pth')))
        
        # Evaluation phase
        # For one subject, averge 5 output probabilities using 5 random combinations of gesture cycles.

        kinematic_model.eval()
        # Define datasets for the current fold.
        test_set = TimeSeriesDataset(
                        data={key: data[key] for key in all_sc[test_idx] if key in valid_sc},
                        gt={key: gt_labels[key] for key in all_sc[test_idx] if key in valid_sc},
                        gesture_list=configs['gesture_list'], 
                        target_cycle_len=configs['target_cycle_len'], 
                        is_train=False,
                        mean=train_set.mean, 
                        std=train_set.std,
                        downsample_rate=configs['downsample_rate']
                        )  
        correct = 0  # Counter for correct predictions
        total = 0    # Counter for total samples in the fold
        for idx in range(len(test_set)):
            probs = np.zeros(num_classes)
            total_loss = 0
            for _ in range(5):
                sample, label = test_set[idx] 
                sample = sample.unsqueeze(0).to(device)        
                output = kinematic_model(sample)
                probs += output.squeeze().cpu().detach().numpy()
            # Take the class with the highest average probability
            probs/=5
            final_pred = probs.argmax()
            # Write results to CSV for each test instance
            writer.writerow([fold + 1, test_set.sc_list[idx], final_pred, label.squeeze().item()] + list(probs))

            # Count correct predictions
            total += 1
            if final_pred == label.squeeze().item():
                correct += 1

        # Calculate and print accuracy for the fold
        accuracy = correct / total
        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")

    result_file.close() 

if __name__ == "__main__":  
    main()