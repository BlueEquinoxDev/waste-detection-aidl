import numpy as np
from typing import Callable
import torch
import os

class EarlyStopping:
    def __init__(self, patience=5, delta=0,fn_save_checkpoint:Callable=None,path_dir_save_checkpoint:str=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = np.Inf
        self.stop = False,
        self.fn_save_checkpoint=fn_save_checkpoint                
        self.path_dir_save_checkpoint=path_dir_save_checkpoint

    def __call__(self, val_loss, model,epoch):
        if val_loss < self.best_loss - self.delta:
            self.stop = False
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model,epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def save_checkpoint(self, model,epoch):
        '''Saves model when validation loss decreases.'''
        checkpoint = {
            "model_state_dict":  model.state_dict()
        }  
        if self.fn_save_checkpoint==None:
            if self.path_dir_save_checkpoint==None:
                torch.save(checkpoint, f'early_stoping_checkpoint_{epoch}.pt')
            else:                
                torch.save(checkpoint, os.path.join(self.path_dir_save_checkpoint,f'early_stoping_checkpoint_{epoch}.pt'))
        else:
            self.fn_save_checkpoint(model,epoch)