import os

import torch
import numpy as np
import torch.nn as nn

from utils.config import DEFAULT_CHECKPOINTS_NAME
from utils.logger import logger

class EarlyStopping:
    def __init__(self, model:nn.Module, patience:int, save_path:str, delta:float=0):
        self.model = model
        self.patience = patience
        self.save_path = save_path
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.stop = False
        self.val_loss_min = np.inf

    def update(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss:float) -> None:
        logger.info(f'validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(self.model.state_dict(), os.path.join(self.save_path, DEFAULT_CHECKPOINTS_NAME))
        self.val_loss_min = val_loss
