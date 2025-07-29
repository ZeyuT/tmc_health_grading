import numpy as np
import os
import sys
import random 

import csv
import math
from scipy import stats

import torch
import torch.nn as nn

import torch.nn.functional as F

def test(model, reference_set, test_set, writer, fold, device, top_k=1):
    model.eval()
    
    # Embed reference samples using training data
    # Each training subject contributes one sample
    ref_samples, ref_labels, hid_list = reference_set.get_data()
    with torch.no_grad():
        ref_embeddings = model(ref_samples.to(device))
    total = 0
    correct = 0
    for idx in range(len(test_set)):
        total_loss = 0
        sample, label = test_set[idx] 
        sample = sample.unsqueeze(0).to(device)       
        with torch.no_grad():
            test_embedding = model(sample)
        distances = F.pairwise_distance(test_embedding, ref_embeddings)
        # Take major class of three training sample with min distance 
        values, min_indices = torch.topk(distances, k=top_k, largest=False)
        min_indices = min_indices.tolist()
        values = values.tolist()
        pred = np.argmax(np.bincount(ref_labels[min_indices]))
        
        # Write results to CSV for each test instance
        writer.writerow([fold + 1, test_set.hid_list[idx], pred, label.squeeze().item()]+\
                        [f'{hid_list[min_indices[i]]} ({ref_labels[min_indices[i]]}) {values[i]:.04f}' for i in range(top_k)])

        # Count correct predictions
        total += 1
        if pred == label.squeeze().item():
            correct += 1

    # Calculate and print accuracy for the fold
    accuracy = correct / total
    print(f"Fold {fold + 1} Accuracy: {accuracy * 100:.2f}%")