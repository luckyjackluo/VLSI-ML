import torch
from torch import nn
import os
import numpy as np
import torch.nn.functional as F


def train(n, train_dataset, val_dataset, model):
    for epoch in range(n):
        model.train()

        loss_train = 0
        loss_val = 0
        correct_train = 0
        correct_val = 0

        for data in train_dataset:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch, data.eig, data.stats)[0]
            pred = output.argmax(dim=0)  
            label = data.y.argmax(dim=0)  
            correct_train += int(pred==label)
            loss = F.cross_entropy(output, data.y)
            loss.backward()
            loss_train += loss.item()
            optimizer.step()

        for data in val_dataset:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch, data.eig, data.stats)[0]
            pred = output.argmax(dim=0)  
            label = data.y.argmax(dim=0)  
            correct_val += int(pred==label)
            loss_t = F.cross_entropy(output, data.y)
            loss_val += loss_t.item()


        loss_train = loss_train/len(train_dataset)
        loss_val = loss_val/len(val_dataset)
        acc_train = correct_train/len(train_dataset)
        acc_val = correct_val/len(val_dataset)

        f.write(f"{epoch}, {loss_train}, {loss_val}, {acc_train}, {acc_val}\n")

        if epoch//100 > 0 and epoch%100 == 0:
            torch.save(model, "trained.pt")
