import os
import torch
import numpy as np
import optuna
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from scripts.train_resnet_classification_opt import label_names, train_loader, val_loader

def save_hparams(hparams, filename="./utilities/hparams.json"):
    with open(filename, "w") as f:
        json.dump(hparams, f)

def load_hparams(filename="./utilities/hparams.json"):
    with open(filename, "r") as f:
        return json.load(f)

def objective(trial):
    hparams = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "dropout": trial.suggest_uniform("dropout", 0.2, 0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    }
    
    save_hparams(hparams) 

    model = models.resnet50(pretrained=True)
    num_classes = len(label_names)
    model.fc = nn.Sequential(
        nn.Dropout(hparams["dropout"]),
        nn.Linear(model.fc.in_features, num_classes)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if hparams["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])
    elif hparams["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=hparams["lr"], momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=hparams["lr"])

    num_epochs = 10
    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_train_loss = 0.0
        correct_preds, total_preds = 0, 0

        for batch in tqdm(train_loader):
            images, labels = batch['pixel_values'].to(device), batch['labels'].to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct_preds += (outputs.argmax(1) == labels).sum().item()
            total_preds += labels.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        train_losses.append(train_loss)

        model.eval()
        val_loss, correct_preds, total_preds, running_val_loss = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch['pixel_values'].to(device), batch['labels'].to(device).long()
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                correct_preds += (outputs.argmax(1) == labels).sum().item()
                total_preds += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        val_losses.append(val_loss)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

    trial.set_user_attr("train_loss", train_losses[-1])
    trial.set_user_attr("val_loss", val_losses[-1])

    return best_val_acc

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=20)
