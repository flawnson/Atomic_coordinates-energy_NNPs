# TODO: Set up per-atom type dataloader
#       Build model checkpoint retriever function

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy

from torch.utils.data import DataLoader
from ANI1_dataset_master.readers import pyanitools as pya
from __future__ import print_function, division
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

plt.ion()   # interactive mode

"""IMPORT DATA"""
# dataloader = 

"""FEEDFORWARD NN MODEL"""
class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(288, 144)
        self.fc2 = nn.Linear(144, 72)
        self.fc3 = nn.Linear(72, 18)
        self.fc4 = nn.Linear(18, 1)

        self.dropout = nn.Dropout(p=0.30)

    def forward(self, x):

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        # x = F.CELU(self.fc1(x))
        # x = F.CELU(self.fc2(x))
        # x = F.CELU(self.fc3(x))
        x = F.CELU(self.fc4(x))

        return x

# Construct the data loader class
def train(model, criterion, optimizer, scheduler, num_epochs):
    prior = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Epoch set to either train or validation
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward fuction only tracks progress model in train mode
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))

            if phase == "train" and epoch_acc > best_acc:
                torch.save({
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss,
                            "acc": epoch_acc,
                }, r"C:\Users\Flawnson\env\Models\weights-improvement-{epoch:02d}-{loss:.4f}.pt")

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print("\n")

    time_elapsed = time.time() - prior
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
regressor = Regression().to(device)
# num_epochs = 
# criterion = 