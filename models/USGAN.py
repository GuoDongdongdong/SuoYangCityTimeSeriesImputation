from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BRITS import BackboneBRITS
from models.BaseImpuateModel import BaseImputeModel
from utils.functions import calc_mse
from utils.ExperimentArgs import ExperimentArgs


class Model(BaseImputeModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.exp_args = exp_args
        self.n_steps = self.exp_args['lookback_length']
        self.n_features = len(self.exp_args['targets'])
        self.rnn_hidden_size = self.exp_args['rnn_hidden_size']
        self.lambda_mse = self.exp_args['lambda_mse']
        self.hint_rate = self.exp_args['hint_rate']
        self.dropout_rate = self.exp_args['dropout']
        self.model = _USGAN(
            self.n_steps,
            self.n_features,
            self.rnn_hidden_size,
            self.lambda_mse,
            self.hint_rate,
            self.dropout_rate,
        )

    def get_generator(self) -> nn.Module:
        return self.model.backbone.generator
    
    def get_discriminator(self) -> nn.Module:
        return self.model.backbone.discriminator
    
    def _get_inputs(self, batch:dict) -> dict: 
        res = {
            'forward':{
                'X'            : batch['observed_data'],
                'missing_mask' : batch['ground_truth_mask'],
                'deltas'       : batch['deltas']
            },
            'backward' : {
                'X'            : batch['observed_data'],
                'missing_mask' : batch['ground_truth_mask'],
                'deltas'       : batch['deltas']
            }
        }
        return res
    
    def evaluate(self, batch:dict, training:bool, object:str="generator") -> torch.Tensor:
        inputs = self._get_inputs(batch)
        res = self.model.forward(inputs, object, training)
        return res['loss']
    
    def impute(self, batch) -> torch.Tensor:
        inputs = self._get_inputs(batch)
        res = self.model.forward(inputs, "generator", False)
        # imputed_data = res['imputed_data'][0]
        return res['imputed_data']
    

class _USGAN(nn.Module):
    """USGAN model"""

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        lambda_mse: float,
        hint_rate: float = 0.7,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.backbone = BackboneUSGAN(
            n_steps,
            n_features,
            rnn_hidden_size,
            lambda_mse,
            hint_rate,
            dropout_rate,
        )

    def forward(
        self,
        inputs: dict,
        training_object: str = "generator",
        training: bool = True,
    ) -> dict:
        assert training_object in [
            "generator",
            "discriminator",
        ], 'training_object should be "generator" or "discriminator"'
        results = {}
        if training_object == "discriminator":
            imputed_data, discrimination_loss = self.backbone(
                inputs, training_object, training
            )
            loss = discrimination_loss
        else:
            imputed_data, generation_loss = self.backbone(
                inputs,
                training_object,
                training,
            )
            loss = generation_loss
        results["loss"] = loss
        results["imputed_data"] = imputed_data
        return results


class UsganDiscriminator(nn.Module):
    """model Discriminator: built on BiRNN

    Parameters
    ----------
    n_features :
        the feature dimension of the input

    rnn_hidden_size :
        the hidden size of the RNN cell

    hint_rate :
        the hint rate for the input imputed_data

    dropout_rate :
        the dropout rate for the output layer

    device :
        specify running the model on which device, CPU/GPU

    """

    def __init__(
        self,
        n_features: int,
        rnn_hidden_size: int,
        hint_rate: float = 0.7,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.hint_rate = hint_rate
        self.biRNN = nn.GRU(
            n_features * 2, rnn_hidden_size, bidirectional=True, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.read_out = nn.Linear(rnn_hidden_size * 2, n_features)

    def forward(
        self,
        imputed_X: torch.Tensor,
        missing_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward processing of USGAN Discriminator.

        Parameters
        ----------
        imputed_X : torch.Tensor,
            The original X with missing parts already imputed.

        missing_mask : torch.Tensor,
            The missing mask of X.

        Returns
        -------
        logits : torch.Tensor,
            the logits of the probability of being the true value.

        """

        device = imputed_X.device
        hint = (
            torch.rand_like(missing_mask, dtype=torch.float, device=device)
            < self.hint_rate
        )
        hint = hint.int()
        h = hint * missing_mask + (1 - hint) * 0.5
        x_in = torch.cat([imputed_X, h], dim=-1)

        out, _ = self.biRNN(x_in)
        logits = self.read_out(self.dropout(out))
        return logits


class BackboneUSGAN(nn.Module):
    """USGAN model"""

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
        lambda_mse: float,
        hint_rate: float = 0.7,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse

        self.generator = BackboneBRITS(n_steps, n_features, rnn_hidden_size)
        self.discriminator = UsganDiscriminator(
            n_features,
            rnn_hidden_size,
            hint_rate,
            dropout_rate,
        )

    def forward(
        self,
        inputs: dict,
        training_object: str = "generator",
        training: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            _,
            _,
            _,
            _,
        ) = self.generator(inputs)

        # if in training mode, return results with losses
        forward_X = inputs["forward"]["X"]
        forward_missing_mask = inputs["forward"]["missing_mask"]

        if training_object == "discriminator":
            discrimination = self.discriminator(
                imputed_data.detach(), forward_missing_mask
            )
            l_D = F.binary_cross_entropy_with_logits(
                discrimination, forward_missing_mask
            )
            discrimination_loss = l_D
            return imputed_data, discrimination_loss
        else:
            discrimination = self.discriminator(imputed_data, forward_missing_mask)
            l_G = -F.binary_cross_entropy_with_logits(
                discrimination,
                forward_missing_mask,
                weight=1 - forward_missing_mask,
            )
            reconstruction = (f_reconstruction + b_reconstruction) / 2
            reconstruction_loss = calc_mse(
                forward_X, reconstruction, forward_missing_mask
            ) + 0.1 * calc_mse(f_reconstruction, b_reconstruction)
            loss_gene = l_G + self.lambda_mse * reconstruction_loss
            generation_loss = loss_gene
            return imputed_data, generation_loss
