import numpy as np
import os
import sys

import pandas as pd
import imageio
import csv
import glob
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch import optim

from torchvision import transforms
from torchinfo import summary

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score

from data_loader import ImageDataset
from model import SwinUNETRClassifier


def main():    
    # Configurations
    config_path = '../configs.yaml'
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
        
    root_path =  configs['root_path']
    raw_data_path = configs['raw_data_path']
    
    gt_path = configs['gt_path']
    results_path = os.path.join(configs['results_path'], 'image_model')
    os.makedirs(results_path, exist_ok=True)
    
    model_configs = configs['image_model_configs']
    gt_type = model_configs['gt_type']

    cv_configs = configs['cv_configs']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ground truth labels and data
    with open(gt_path, 'rb') as file:
        grades = pickle.load(file)    
        
    gt_labels = grades[gt_type]
                
    all_sc = list(gt_labels.keys())
    
    # Identify sample code which has CT image data.
    valid_sc = []
    for sc in all_sc:
        image_list = glob.glob(os.path.join(configs['raw_data_path'], f"{sc}/BonyGeometry/DICOMs/CT*/*.IMA"))
        if len(image_list) > 0:
            valid_sc.append(sc)
    
    num_classes = len(np.unique(list(gt_labels.values())))
    kf = KFold(n_splits=cv_configs['num_splits'], 
               shuffle=True, 
               random_state=cv_configs['random_seed'])
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizes the tensor to [-1, 1]
    ])
 
    # Open a CSV file to write the results
    result_file = open(os.path.join(results_path, f'result_{gt_type}.csv'), mode='w', newline='')
    writer = csv.writer(result_file)
    writer.writerow(['Fold', 'Sample Code', 'Prediction', 'Ground Truth']+[f'P{class_}' for class_ in range(num_classes)])

    # Main loop for K-fold Cross Validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_sc)):
        print(f"Fold {fold + 1}/{cv_configs['num_splits']}")

        # Define datasets for the current fold
        train_set = ImageDataset(
            gt={key: gt_labels[key] for key in np.array(all_sc)[train_idx] if key in valid_sc},
            data_path=os.path.join(configs['processed_data_path'], 'images'),
        )

        # Create data loaders for the current fold
        train_loader = DataLoader(train_set, 
                                  batch_size=model_configs['batch_size'], 
                                  shuffle=True, 
                                  num_workers=model_configs['num_workers'], pin_memory=True)

        # Model setup for the current fold
        unet = SwinUNETRClassifier(model_configs['seed_path'], num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Adam(unet.parameters(), 
                         lr=model_configs['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=model_configs['patience'], 
                                                         verbose=True)

        # Training phase
        unet.train()
        best_loss = 10000
        for epoch in range(model_configs['num_epochs']):
            total_loss = 0
            steps = 0
            corrects = 0
            total_samples = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = unet(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, preds = torch.max(outputs, 1)
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
                torch.save(unet.state_dict(), os.path.join(results_path, f'model_{gt_type}_fold_{fold+1}.pth'))      
                best_loss = train_loss
                
        # Load weights from the saved best model
        unet.load_state_dict(torch.load(os.path.join(results_path, f'model_{gt_type}_fold_{fold+1}.pth')))
        
        # Evaluation phase
        unet.eval()
        # Define datasets for the current fold
        test_set = ImageDataset(
            gt={key: gt_labels[key] for key in np.array(all_sc)[test_idx] if key in valid_sc},
            data_path=os.path.join(configs['processed_data_path'], 'images'),
        )
        
        correct = 0  # Counter for correct predictions
        total = 0    # Counter for total samples in the fold
        for idx in range(len(test_set)):
            image, label = test_set[idx] 
            image = image.unsqueeze(0).to(device)        
            output = unet(image)
            _, pred = torch.max(output, 1)
            
            # Write results to CSV for each test instance
            writer.writerow([fold + 1, test_set.sc_list[idx], pred.item(), label.item()] + output.squeeze().tolist())

            # Count correct predictions
            total += 1
            if pred.item() == label.item():
                correct += 1

        # Calculate and print accuracy for the fold
        accuracy = correct / total
        print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")
        sys.stdout.flush()
    result_file.close() 

if __name__ == "__main__":  
    main()