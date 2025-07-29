import numpy as np
import os
import sys
import random 

import csv
import math

import torch
import torch.nn as nn
from torch.optim import Adam
from torch import optim

import torch.nn.functional as F


def train(train_loader, model, model_configs, model_path, device):
        optimizer = Adam(model.parameters(), 
                     lr=model_configs['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                         mode='min', 
                                                         factor=0.5, 
                                                         patience=model_configs['patience'], 
                                                         verbose=True)
        model.train()
        best_loss = 10000
        for epoch in range(model_configs['num_epochs']):
            total_loss = 0
            steps = 0
            for sample1, sample2, labels in train_loader:
                optimizer.zero_grad()
                labels = labels.to(device)
                output1 = model(sample1.to(device))
                output2 = model(sample2.to(device))

                loss = contrastive_loss(output1, output2, labels, margin=model_configs['constrastive_margin'])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                if steps > model_configs['max_step']:
                    break
            train_loss = total_loss / steps
            scheduler.step(train_loss)
            print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")
            sys.stdout.flush()
            
            if best_loss > train_loss:               
                # Save the best model
                torch.save(model.state_dict(), model_path)    
                best_loss = train_loss
        return model
    
def contrastive_loss(x1, x2, label, margin=1.0):
    # Use Siamese networks
    # label: 1 for similar, 0 for dissimilar

    # L2 distance
    l2_distance = F.pairwise_distance(x1, x2)
    
    return torch.mean(label * l2_distance**2 + (1 - label) * F.relu(margin - l2_distance)**2)

