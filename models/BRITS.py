import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from models.BaseImpuateModel import BaseImputeModel
from utils.ExperimentArgs import ExperimentArgs
from utils.functions import calc_mae


class Model(BaseImputeModel):

    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.n_steps = exp_args['lookback_length']
        self.n_features = len(exp_args['targets'])
        self.rnn_hidden_size = exp_args['rnn_hidden_size']
        self.model = _BRITS(self.n_steps, self.n_features, self.rnn_hidden_size)

    def _get_input(self, batch:dict) -> dict:
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

    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        inputs = self._get_input(batch)
        res = self.model.forward(inputs, training)
        return res['loss']

    def impute(self, batch:dict) -> torch.Tensor:
        inputs = self._get_input(batch)
        res = self.model.forward(inputs, False)
        return res['imputed_data']


class _BRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.model = BackboneBRITS(n_steps, n_features, rnn_hidden_size)

    def forward(self, inputs: dict, training: bool = True) -> dict:
        (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        ) = self.model(inputs)

        results = {
            "imputed_data": imputed_data,
        }

        results["consistency_loss"] = consistency_loss
        results["reconstruction_loss"] = reconstruction_loss
        loss = consistency_loss + reconstruction_loss
        results["loss"] = loss
        results["reconstruction"] = (f_reconstruction + b_reconstruction) / 2
        results["f_reconstruction"] = f_reconstruction
        results["b_reconstruction"] = b_reconstruction

        return results


class FeatureRegression(nn.Module):
    """The module used to capture the correlation between features for imputation in BRITS.

    Attributes
    ----------
    W : tensor
        The weights (parameters) of the module.

    b : tensor
        The bias of the module.

    m (buffer) : tensor
        The mask matrix, a squire matrix with diagonal entries all zeroes while left parts all ones.
        It is applied to the weight matrix to mask out the estimation contributions from features themselves.
        It is used to help enhance the imputation performance of the network.

    Parameters
    ----------
    input_size : the feature dimension of the input
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the NN module.

        Parameters
        ----------
        x : tensor,
            the input for processing

        Returns
        -------
        output: tensor,
            the processed result containing imputation from feature regression

        """
        output = F.linear(x, self.W * Variable(self.m), self.b)
        return output


class TemporalDecay(nn.Module):
    """The module used to generate the temporal decay factor gamma in the GRU-D model.
    Please refer to the original paper :cite:`che2018GRUD` for more details.

    Attributes
    ----------
    W: tensor,
        The weights (parameters) of the module.
    b: tensor,
        The bias of the module.

    Parameters
    ----------
    input_size : int,
        the feature dimension of the input

    output_size : int,
        the feature dimension of the output

    diag : bool,
        whether to product the weight with an identity matrix before forward processing

    References
    ----------
    .. [1] `Che, Zhengping, Sanjay Purushotham, Kyunghyun Cho, David Sontag, and Yan Liu.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8, no. 1 (2018): 6085.
        <https://www.nature.com/articles/s41598-018-24271-9.pdf>`_

    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super().__init__()
        self.diag = diag
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag:
            assert input_size == output_size
            m = torch.eye(input_size, input_size)
            self.register_buffer("m", m)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std_dev = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-std_dev, std_dev)
        if self.b is not None:
            self.b.data.uniform_(-std_dev, std_dev)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Forward processing of this NN module.

        Parameters
        ----------
        delta : tensor, shape [n_samples, n_steps, n_features]
            The time gaps.

        Returns
        -------
        gamma : tensor, of the same shape with parameter `delta`, values in (0,1]
            The temporal decay factor.
        """
        if self.diag:
            gamma = F.relu(F.linear(delta, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(delta, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class BackboneRITS(nn.Module):
    """model RITS: Recurrent Imputation for Time Series

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rnn_cell :
        the LSTM cell to model temporal data

    temp_decay_h :
        the temporal decay module to decay RNN hidden state

    temp_decay_x :
        the temporal decay module to decay data in the raw feature space

    hist_reg :
        the temporal-regression module to project RNN hidden state into the raw feature space

    feat_reg :
        the feature-regression module

    combining_weight :
        the module used to generate the weight to combine history regression and feature regression

    Parameters
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell


    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size

        self.rnn_cell = nn.LSTMCell(self.n_features * 2, self.rnn_hidden_size)
        self.temp_decay_h = TemporalDecay(
            input_size=self.n_features, output_size=self.rnn_hidden_size, diag=False
        )
        self.temp_decay_x = TemporalDecay(
            input_size=self.n_features, output_size=self.n_features, diag=True
        )
        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.n_features)
        self.feat_reg = FeatureRegression(self.n_features)
        self.combining_weight = nn.Linear(self.n_features * 2, self.n_features)

    def forward(
        self, inputs: dict, direction: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs :
            Input data, a dictionary includes feature values, missing masks, and time-gap values.

        direction :
            A keyword to extract data from `inputs`.

        Returns
        -------
        imputed_data :
            Input data with missing parts imputed. Shape of [batch size, sequence length, feature number].

        estimations :
            Reconstructed data. Shape of [batch size, sequence length, feature number].

        hidden_states: tensor,
            [batch size, RNN hidden size]

        reconstruction_loss :
            reconstruction loss

        """
        X = inputs[direction]["X"]  # feature values
        missing_mask = inputs[direction]["missing_mask"]  # mask marks missing part in X
        deltas = inputs[direction]["deltas"]  # time-gap values

        device = X.device

        # create hidden states and cell states for the lstm cell
        hidden_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)
        cell_states = torch.zeros((X.size()[0], self.rnn_hidden_size), device=device)

        estimations = []
        reconstruction_loss = torch.tensor(0.0).to(device)

        # imputation period
        for t in range(self.n_steps):
            # data shape: [batch, time, features]
            x = X[:, t, :]  # values
            m = missing_mask[:, t, :]  # mask
            d = deltas[:, t, :]  # delta, time gap

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            hidden_states = hidden_states * gamma_h  # decay hidden states
            x_h = self.hist_reg(hidden_states)
            reconstruction_loss += calc_mae(x_h, x, m)

            x_c = m * x + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            reconstruction_loss += calc_mae(z_h, x, m)

            alpha = torch.sigmoid(self.combining_weight(torch.cat([gamma_x, m], dim=1)))

            c_h = alpha * z_h + (1 - alpha) * x_h
            reconstruction_loss += calc_mae(c_h, x, m)

            c_c = m * x + (1 - m) * c_h
            estimations.append(c_h.unsqueeze(dim=1))

            inputs = torch.cat([c_c, m], dim=1)
            hidden_states, cell_states = self.rnn_cell(
                inputs, (hidden_states, cell_states)
            )

        # for each iteration, reconstruction_loss increases its value for 3 times
        reconstruction_loss /= self.n_steps * 3

        reconstruction = torch.cat(estimations, dim=1)
        imputed_data = missing_mask * X + (1 - missing_mask) * reconstruction

        return imputed_data, reconstruction, hidden_states, reconstruction_loss


class BackboneBRITS(nn.Module):
    """model BRITS: Bidirectional RITS
    BRITS consists of two RITS, which take time-series data from two directions (forward/backward) respectively.

    Attributes
    ----------
    n_steps :
        sequence length (number of time steps)

    n_features :
        number of features (input dimensions)

    rnn_hidden_size :
        the hidden size of the RNN cell

    rits_f: RITS object
        the forward RITS model

    rits_b: RITS object
        the backward RITS model

    """

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        rnn_hidden_size: int,
    ):
        super().__init__()
        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        # create models
        self.rits_f = BackboneRITS(n_steps, n_features, rnn_hidden_size)
        self.rits_b = BackboneRITS(n_steps, n_features, rnn_hidden_size)

    @staticmethod
    def _get_consistency_loss(
        pred_f: torch.Tensor, pred_b: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f :
            The imputation from the forward RITS.

        pred_b :
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.

        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    @staticmethod
    def _reverse(ret: Tuple) -> Tuple:
        """Reverse the array values on the time dimension in the given dictionary."""

        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        collector = []
        for value in ret:
            collector.append(reverse_tensor(value))

        return tuple(collector)

    def forward(self, inputs: dict) -> Tuple[torch.Tensor, ...]:
        # Results from the forward RITS.
        (
            f_imputed_data,
            f_reconstruction,
            f_hidden_states,
            f_reconstruction_loss,
        ) = self.rits_f(inputs, "forward")
        # Results from the backward RITS.
        (
            b_imputed_data,
            b_reconstruction,
            b_hidden_states,
            b_reconstruction_loss,
        ) = self._reverse(self.rits_b(inputs, "backward"))

        imputed_data = (f_imputed_data + b_imputed_data) / 2
        consistency_loss = self._get_consistency_loss(f_imputed_data, b_imputed_data)
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss

        return (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        )
