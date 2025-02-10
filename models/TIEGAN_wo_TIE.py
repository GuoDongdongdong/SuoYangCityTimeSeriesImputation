'''
TIEGAN without time information enhancement block.
use for chapter 3 experiment 3 table 3.3 third column.
'''

from typing import Callable
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ExperimentArgs import ExperimentArgs
from models.BaseImpuateModel import BaseImputeModel
from models.SAITS import _SAITS
from utils.functions import calc_mae


from typing import Callable, Optional, Tuple


class Model(BaseImputeModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.exp_args = exp_args
        self.n_layers = self.exp_args['n_layers']
        self.n_steps = self.exp_args['lookback_length']
        self.n_features = len(self.exp_args['targets'])
        self.d_model = self.exp_args['d_model']
        self.n_heads = self.exp_args['n_heads']
        self.d_k = self.exp_args['d_k']
        self.d_v = self.exp_args['d_v']
        self.d_ffn = self.exp_args['d_ffn']
        self.dropout = self.exp_args['dropout']
        self.attn_dropout = self.exp_args['attn_dropout']
        self.diagonal_attention_mask = self.exp_args['diagonal_attention_mask']
        self.ORT_weight = self.exp_args['ort_weight']
        self.MIT_weight = self.exp_args['mit_weight']
        self.rnn_hidden_size = self.exp_args['rnn_hidden_size']
        self.hint_rate = self.exp_args['hint_rate']
        self.discriminator_dropout_rate = self.exp_args['discriminator_dropout_rate']
        self.date_frequence = self.exp_args['date_frequence']
        self.temporal_dim = self.exp_args['temporal_dim']
        self.model = _TIEGAN(
            self.n_layers,
            self.n_steps,
            self.n_features,
            self.d_model,
            self.n_heads,
            self.d_k,
            self.d_v,
            self.d_ffn,
            self.dropout,
            self.attn_dropout,
            self.rnn_hidden_size,
            self.hint_rate,
            self.discriminator_dropout_rate,
            self.diagonal_attention_mask,
            self.ORT_weight,
            self.MIT_weight,
            self.date_frequence,
            self.temporal_dim,
        )

    def _get_input(self, batch:dict) -> dict:
        res = {
            'X' : batch['observed_data'],
            'observed_mask' : batch['ground_truth_mask'],
            'deltas' : batch['deltas'],
            'date' : batch['date'],
            'missing_mask' : batch['ground_truth_mask'],
            'X_ori' : batch['observed_data'],
            'indicating_mask' : batch['loss_mask']
        }
        return res
    
    def get_generator(self) -> nn.Module:
        return self.model.generator
    
    def get_discriminator(self) -> nn.Module:
        return self.model.discriminator

    def evaluate(self, batch:dict, training:bool, object:str="generator") -> torch.Tensor:
        inputs = self._get_input(batch)
        res = self.model.forward(inputs, object, training)
        return res['loss']

    def impute(self, batch:dict) -> torch.Tensor:
        inputs = self._get_input(batch)
        res = self.model.forward(inputs, "generator", False)
        return res['imputed_data']


class _TIEGAN(nn.Module):
    def __init__(self,
            n_layers: int,
            n_steps: int,
            n_features: int,
            d_model: int,
            n_heads: int,
            d_k: int,
            d_v: int,
            d_ffn: int,
            dropout: float,
            attn_dropout: float,
            rnn_hidden_size: int,
            hint_rate: float = 0.7,
            discrimintor_dropout_rate: float = 0.0,
            diagonal_attention_mask: bool = True,
            ORT_weight: float = 1,
            MIT_weight: float = 1,
            date_frequence: str = 's',
            temporal_dim: int = 64,
            customized_loss_func: Callable = calc_mae,
        ):
        super().__init__()
        self.generator = _SAITS(
            n_layers,
            n_steps,
            n_features,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
            diagonal_attention_mask,
            ORT_weight,
            MIT_weight,
            customized_loss_func,
            date_frequence,
            temporal_dim,
        )
        self.discriminator = Discriminator(
            n_features,
            rnn_hidden_size,
            hint_rate,
            discrimintor_dropout_rate,
        )
        

    def forward(self, inputs:dict, training_object:str='generator', training:bool=True):
        assert training_object in ['generator', 'discriminator'], 'training_object should be generator or discriminator!'
        res = self.generator(inputs, training=training)
        imputed_data = res['imputed_data']
        observed_mask = inputs['observed_mask']
        if training_object == "discriminator":
            discrimination = self.discriminator(
                imputed_data.detach(), observed_mask
            )
            l_D = F.binary_cross_entropy_with_logits(
                discrimination, observed_mask
            )
            discrimination_loss = l_D
            res['loss'] = discrimination_loss
        return res
    

class Discriminator(nn.Module):
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
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward processing of USGAN Discriminator.

        Parameters
        ----------
        imputed_X : torch.Tensor,
            The original X with missing parts already imputed.

        observed_mask : torch.Tensor,
            The missing mask of X.

        Returns
        -------
        logits : torch.Tensor,
            the logits of the probability of being the true value.

        """

        device = imputed_X.device
        hint = (
            torch.rand_like(observed_mask, dtype=torch.float, device=device)
            < self.hint_rate
        )
        hint = hint.int()
        h = hint * observed_mask + (1 - hint) * 0.5
        x_in = torch.cat([imputed_X, h], dim=-1)

        out, _ = self.biRNN(x_in)
        logits = self.read_out(self.dropout(out))
        return logits


class _SAITS(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_steps: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        diagonal_attention_mask: bool = True,
        ORT_weight: float = 1,
        MIT_weight: float = 1,
        customized_loss_func: Callable = calc_mae,
        date_frequence: str = 's',
        temporal_dim: int = 64,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.diagonal_attention_mask = diagonal_attention_mask
        self.ORT_weight = ORT_weight
        self.MIT_weight = MIT_weight
        self.customized_loss_func = customized_loss_func

        self.encoder = BackboneSAITS(
            n_steps,
            n_features,
            n_layers,
            d_model,
            n_heads,
            d_k,
            d_v,
            d_ffn,
            dropout,
            attn_dropout,
            date_frequence,
            temporal_dim,
        )

    def forward(
        self,
        inputs: dict,
        diagonal_attention_mask: bool = True,
        training: bool = True,
    ) -> dict:
        X, observed_mask = inputs["X"], inputs["observed_mask"]
        date, deltas = inputs['date'], inputs['deltas']
        # determine the attention mask
        if (training and self.diagonal_attention_mask) or (
            (not training) and diagonal_attention_mask
        ):
            diagonal_attention_mask = (1 - torch.eye(self.n_steps)).to(X.device)
            # then broadcast on the batch axis
            diagonal_attention_mask = diagonal_attention_mask.unsqueeze(0)
        else:
            diagonal_attention_mask = None

        # SAITS processing
        (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        ) = self.encoder(X, observed_mask, date, deltas, diagonal_attention_mask)

        # replace the observed part with values from X
        imputed_data = observed_mask * X + (1 - observed_mask) * X_tilde_3

        # ensemble the results as a dictionary for return
        results = {
            "first_DMSA_attn_weights": first_DMSA_attn_weights,
            "second_DMSA_attn_weights": second_DMSA_attn_weights,
            "combining_weights": combining_weights,
            "imputed_data": imputed_data,
        }

        X_ori, indicating_mask = inputs["X_ori"], inputs["indicating_mask"]

        # calculate loss for the observed reconstruction task (ORT)
        # this calculation is more complicated that pypots.nn.modules.saits.SaitsLoss because
        # SAITS model structure has three parts of representation
        ORT_loss = 0
        ORT_loss += self.customized_loss_func(X_tilde_1, X, observed_mask)
        ORT_loss += self.customized_loss_func(X_tilde_2, X, observed_mask)
        ORT_loss += self.customized_loss_func(X_tilde_3, X, observed_mask)
        ORT_loss /= 3
        ORT_loss = self.ORT_weight * ORT_loss

        # calculate loss for the masked imputation task (MIT)
        MIT_loss = self.MIT_weight * self.customized_loss_func(
            X_tilde_3, X_ori, indicating_mask
        )
        # `loss` is always the item for backward propagating to update the model
        loss = ORT_loss + MIT_loss

        results["ORT_loss"] = ORT_loss
        results["MIT_loss"] = MIT_loss
        results["loss"] = loss

        return results


class PositionalEncoding(nn.Module):
    """The original positional-encoding module for Transformer.

    Parameters
    ----------
    d_hid:
        The dimension of the hidden layer.

    n_positions:
        The max number of positions.

    """

    def __init__(self, d_hid: int, n_positions: int = 1000):
        super().__init__()
        pe = torch.zeros(n_positions, d_hid, requires_grad=False).float()
        position = torch.arange(0, n_positions).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_hid, 2).float()
            * -(torch.log(torch.tensor(10000)) / d_hid)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pos_table", pe)

    def forward(self, x: torch.Tensor, return_only_pos: bool = False) -> torch.Tensor:
        """Forward processing of the positional encoding module.

        Parameters
        ----------
        x:
            Input tensor.

        return_only_pos:
            Whether to return only the positional encoding.

        Returns
        -------
        If return_only_pos is True:
            pos_enc:
                The positional encoding.
        else:
            x_with_pos:
                Output tensor, the input tensor with the positional encoding added.
        """
        pos_enc = self.pos_table[:, : x.size(1)].clone().detach()

        if return_only_pos:
            return pos_enc

        x_with_pos = x + pos_enc
        return x_with_pos


class SaitsEmbedding(nn.Module):
    """The embedding method from the SAITS paper :cite:`du2023SAITS`.

    Parameters
    ----------
    d_in :
        The input dimension.

    d_out :
        The output dimension.

    with_pos :
        Whether to add positional encoding.

    n_max_steps :
        The maximum number of steps.
        It only works when ``with_pos`` is True.

    dropout :
        The dropout rate.

    """

    def __init__(
        self,
        n_features: int,
        temporal_dim: int,
        d_in: int,
        d_out: int,
        with_pos: bool,
        n_max_steps: int = 1000,
        dropout: float = 0,
        date_frequence: str = 's'
    ):
        super().__init__()
        self.with_pos = with_pos
        self.dropout_rate = dropout

        self.embedding_layer = nn.Linear(d_in, d_out)
        #self.temporal_enhance_block = TemporalEnhanceBlock(n_features, temporal_dim, date_frequence)
        self.position_enc = (
            PositionalEncoding(d_out, n_positions=n_max_steps) if with_pos else None
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, X, observed_mask, date, deltas):
        #temporal_information = self.temporal_enhance_block(date, deltas)
        X = torch.cat([X, observed_mask], dim=2)
        X_embedding = self.embedding_layer(X)
        #X_embedding = torch.cat([temporal_information, X_embedding], dim=2)
        if self.with_pos:
            X_embedding = self.position_enc(X_embedding)
        if self.dropout_rate > 0:
            X_embedding = self.dropout(X_embedding)

        return X_embedding


class AttentionOperator(nn.Module):
    """
    The abstract class for all attention layers.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class ScaledDotProductAttention(AttentionOperator):
    """Scaled dot-product attention.

    Parameters
    ----------
    temperature:
        The temperature for scaling.

    attn_dropout:
        The dropout rate for the attention map.

    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        assert temperature > 0, "temperature should be positive"
        assert attn_dropout >= 0, "dropout rate should be non-negative"
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the scaled dot-product attention.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        output:
            The result of Value multiplied with the scaled dot-product attention map.

        attn:
            The scaled dot-product attention map.

        """
        # q, k, v all have 4 dimensions [batch_size, n_steps, n_heads, d_tensor]
        # d_tensor could be d_q, d_k, d_v

        # transpose for attention dot product: [batch_size, n_heads, n_steps, d_k or d_v]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # dot product q with k.T to obtain similarity
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # apply masking on the attention map, this is optional
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)

        # compute attention score [0, 1], then apply dropout
        attn = F.softmax(attn, dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # multiply the score with v
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Transformer multi-head attention module.

    Parameters
    ----------
    attn_opt:
        The attention operator, e.g. the self-attention proposed in Transformer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    """

    def __init__(
        self,
        attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.attention_operator = attn_opt
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the multi-head attention module.

        Parameters
        ----------
        q:
            Query tensor.

        k:
            Key tensor.

        v:
            Value tensor.

        attn_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].
            0 in attn_mask means values at the according position in the attention map will be masked out.

        Returns
        -------
        v:
            The output of the multi-head attention layer.

        attn_weights:
            The attention map.

        """
        # the shapes of q, k, v are the same [batch_size, n_steps, d_model]

        batch_size, q_len = q.size(0), q.size(1)
        k_len = k.size(1)
        v_len = v.size(1)

        # now separate the last dimension of q, k, v into different heads -> [batch_size, n_steps, n_heads, d_k or d_v]
        q = self.w_qs(q).view(batch_size, q_len, self.n_heads, self.d_k)
        k = self.w_ks(k).view(batch_size, k_len, self.n_heads, self.d_k)
        v = self.w_vs(v).view(batch_size, v_len, self.n_heads, self.d_v)
        # for generalization, we don't do transposing here but leave it for the attention operator if necessary

        if attn_mask is not None:
            # broadcasting on the head axis
            attn_mask = attn_mask.unsqueeze(1)

        v, attn_weights = self.attention_operator(q, k, v, attn_mask, **kwargs)

        # transpose back -> [batch_size, n_steps, n_heads, d_v]
        # then merge the last two dimensions to combine all the heads -> [batch_size, n_steps, n_heads*d_v]
        v = v.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        v = self.fc(v)

        return v, attn_weights


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed forward network (FFN) in Transformer.

    Parameters
    ----------
    d_in:
        The dimension of the input tensor.

    d_hid:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid)
        self.linear_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward processing of the position-wise feed forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        x:
            Output tensor.
        """
        # save the original input for the later residual connection
        residual = x
        # the 1st linear processing and ReLU non-linear projection
        x = F.relu(self.linear_1(x))
        # the 2nd linear processing
        x = self.linear_2(x)
        # apply dropout
        x = self.dropout(x)
        # apply residual connection
        x += residual
        # apply layer-norm
        x = self.layer_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer.

    Parameters
    ----------
    attn_opt:
        The attention operator for the multi-head attention module in the encoder layer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(
        self,
        attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            attn_opt,
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        enc_input: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward processing of the encoder layer.

        Parameters
        ----------
        enc_input:
            Input tensor.

        src_mask:
            Masking tensor for the attention map. The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        enc_output:
            Output tensor.

        attn_weights:
            The attention map.

        """
        enc_output, attn_weights = self.slf_attn(
            enc_input,
            enc_input,
            enc_input,
            attn_mask=src_mask,
            **kwargs,
        )

        # apply dropout and residual connection
        enc_output = self.dropout(enc_output)
        enc_output += enc_input

        # apply layer-norm
        enc_output = self.layer_norm(enc_output)

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    Parameters
    ----------
    slf_attn_opt:
        The attention operator for the multi-head attention module in the decoder layer.

    enc_attn_opt:
        The attention operator for the encoding multi-head attention module in the decoder layer.

    d_model:
        The dimension of the input tensor.

    n_heads:
        The number of heads in multi-head attention.

    d_k:
        The dimension of the key and query tensor.

    d_v:
        The dimension of the value tensor.

    d_ffn:
        The dimension of the hidden layer.

    dropout:
        The dropout rate.

    """

    def __init__(
        self,
        slf_attn_opt: AttentionOperator,
        enc_attn_opt: AttentionOperator,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.slf_attn = MultiHeadAttention(
            slf_attn_opt,
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.enc_attn = MultiHeadAttention(
            enc_attn_opt,
            d_model,
            n_heads,
            d_k,
            d_v,
        )
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ffn, dropout)

    def forward(
        self,
        dec_input: torch.Tensor,
        enc_output: torch.Tensor,
        slf_attn_mask: Optional[torch.Tensor] = None,
        dec_enc_attn_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward processing of the decoder layer.

        Parameters
        ----------
        dec_input:
            Input tensor.

        enc_output:
            Output tensor from the encoder.

        slf_attn_mask:
            Masking tensor for the self-attention module.
            The shape should be [batch_size, n_heads, n_steps, n_steps].

        dec_enc_attn_mask:
            Masking tensor for the encoding attention module.
            The shape should be [batch_size, n_heads, n_steps, n_steps].

        Returns
        -------
        dec_output:
            Output tensor.

        dec_slf_attn:
            The self-attention map.

        dec_enc_attn:
            The encoding attention map.

        """
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input,
            dec_input,
            dec_input,
            attn_mask=slf_attn_mask,
            **kwargs,
        )
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output,
            enc_output,
            enc_output,
            attn_mask=dec_enc_attn_mask,
            **kwargs,
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class BackboneSAITS(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_features: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_k: int,
        d_v: int,
        d_ffn: int,
        dropout: float,
        attn_dropout: float,
        date_frequence: str,
        temporal_dim: int,
    ):
        super().__init__()

        # concatenate the feature vector and missing mask, hence double the number of features
        actual_n_features = n_features * 2

        # for the 1st block
        self.embedding_1 = SaitsEmbedding(
            n_features,
            temporal_dim,
            actual_n_features,
            d_model,
            with_pos=False,
            n_max_steps=n_steps,
            dropout=dropout,
            date_frequence=date_frequence
        )
        self.layer_stack_for_first_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model + temporal_dim,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.reduce_dim_z = nn.Linear(d_model + temporal_dim, n_features)

        # for the 2nd block
        self.embedding_2 = SaitsEmbedding(
            n_features,
            temporal_dim,
            actual_n_features,
            d_model,
            with_pos=False,
            n_max_steps=n_steps,
            dropout=dropout,
            date_frequence=date_frequence,
        )
        self.layer_stack_for_second_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    ScaledDotProductAttention(d_k**0.5, attn_dropout),
                    d_model + temporal_dim,
                    n_heads,
                    d_k,
                    d_v,
                    d_ffn,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.reduce_dim_beta = nn.Linear(d_model + temporal_dim, n_features)
        self.reduce_dim_gamma = nn.Linear(n_features, n_features)

        # for delta decay factor
        self.weight_combine = nn.Linear(n_features + n_steps, n_features)

    def forward(
        self, X, observed_mask, date, deltas, attn_mask: Optional = None
    ) -> Tuple[torch.Tensor, ...]:

        # first DMSA block
        enc_output = self.embedding_1(
            X, observed_mask, date, deltas
        )  # namely, term e in the math equation
        first_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, first_DMSA_attn_weights = encoder_layer(enc_output, attn_mask)
        X_tilde_1 = self.reduce_dim_z(enc_output)
        X_prime = observed_mask * X + (1 - observed_mask) * X_tilde_1

        # second DMSA block
        enc_output = self.embedding_2(
            X_prime, observed_mask, date, deltas
        )  # namely term alpha in math algo
        second_DMSA_attn_weights = None
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, second_DMSA_attn_weights = encoder_layer(enc_output, attn_mask)
        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        copy_second_DMSA_weights = second_DMSA_attn_weights.clone()
        copy_second_DMSA_weights = copy_second_DMSA_weights.squeeze(
            dim=1
        )  # namely term A_hat in Eq.
        if len(copy_second_DMSA_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 3)
            copy_second_DMSA_weights = copy_second_DMSA_weights.mean(dim=3)
            copy_second_DMSA_weights = torch.transpose(copy_second_DMSA_weights, 1, 2)

        # namely term eta
        combining_weights = torch.sigmoid(
            self.weight_combine(
                torch.cat([observed_mask, copy_second_DMSA_weights], dim=2)
            )
        )
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1

        return (
            X_tilde_1,
            X_tilde_2,
            X_tilde_3,
            first_DMSA_attn_weights,
            second_DMSA_attn_weights,
            combining_weights,
        )


class TemporalEnhanceBlock(nn.Module):
    def __init__(self, n_features:int, temporal_dim:int, date_frequence:str):
        super().__init__()
        date_frequence_dict = {
            'm' : 1,
            'w' : 2,
            'd' : 3,
            'bd' : 3,
            'h' : 4,
            'm' : 5,
            's' : 6
        }
        self.temporal_embedding = nn.Linear(date_frequence_dict[date_frequence] + n_features, temporal_dim)

    def forward(self, date:torch.Tensor, deltas:torch.Tensor):
        temporal_information = torch.cat([date, deltas], dim=2)
        temporal_information = self.temporal_embedding(temporal_information)
        return temporal_information