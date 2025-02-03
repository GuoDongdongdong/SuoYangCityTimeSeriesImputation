from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter


class BaseImputeModel(ABC):
    
    def __init__(self):
        super().__init__()
        self.model = None

    def get_inner_model(self) -> nn.Module | None:
        return self.model

    def get_model_params(self) -> int:
        if isinstance(self.model, nn.Module):
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0

    def get_parameters(self) -> Iterator[Parameter]:
        assert isinstance(self.model, nn.Module), 'model is not nn.Module type!'
        return self.model.parameters()

    def load_model(self, save_path:str) -> None:
        assert isinstance(self.model, nn.Module), 'model is not nn.Module type!'
        self.model.load_state_dict(torch.load(save_path))

    @abstractmethod
    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        raise NotImplementedError

    '''
        note that return imputed_data's shape should be like [B, L, D] or [B, n_samples, L, D]
    '''
    @abstractmethod
    def impute(self, batch:dict) -> torch.Tensor:
        raise NotImplementedError

    def get_generator(self) -> nn.Module | None:
        return None
    
    def get_discriminator(self) -> nn.Module | None:
        return None