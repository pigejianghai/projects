from typing import Any
import torch

class LRScheduler():
    def __init__(self, optimizer, patience=10, min_lr=1e-8, factor=0.1):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=self.patience, 
            factor=self.factor, 
            min_lr=self.min_lr, 
            verbose=True
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early Stopping')
                self.early_stop = True

class EarlyStoppingAUC():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc = None
        self.early_stop = False
    
    def __call__(self, val_auc):
        if self.best_auc == None:
            self.best_auc = val_auc
        elif val_auc - self.best_auc > self.min_delta:
            self.best_auc = val_auc
            self.counter = 0
        elif val_auc - self.best_auc < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early Stopping')
                self.early_stop = True