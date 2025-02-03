import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.BaseImpuateModel import BaseImputeModel
from utils.ExperimentArgs import ExperimentArgs


class Model(BaseImputeModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.exp_args = exp_args
        self.n_features = len(self.exp_args['targets'])
        self.n_layers = self.exp_args['n_layers']
        self.n_heads = self.exp_args['n_heads']
        self.n_channels = self.exp_args['n_channels']
        self.d_time_embedding = self.exp_args['d_time_embedding']
        self.d_feature_embedding = self.exp_args['d_feature_embedding']
        self.d_diffusion_embedding = self.exp_args['d_diffusion_embedding']
        self.is_unconditional = self.exp_args['is_unconditional']
        self.n_diffusion_steps = self.exp_args['n_diffusion_steps']
        self.target_strategy = self.exp_args['target_strategy']
        self.schedule = self.exp_args['schedule']
        self.beta_start = self.exp_args['beta_start']
        self.beta_end = self.exp_args['beta_end']
        self.n_samples = self.exp_args['n_samples']

        self.model = _CSDI(
                self.n_features,
                self.n_layers,
                self.n_heads,
                self.n_channels,
                self.d_time_embedding,
                self.d_feature_embedding,
                self.d_diffusion_embedding,
                self.is_unconditional,
                self.n_diffusion_steps,
                self.schedule,
                self.beta_start,
                self.beta_end
        )

    def _get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def _get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def _get_input(self, batch:dict, trianing:bool) -> dict:
        if trianing:
            if self.target_strategy == "random":
                cond_mask = self._get_randmask(batch['observed_mask'])
            else:
                cond_mask = self._get_hist_mask(batch['observed_mask'])
        else:
            cond_mask = batch['ground_truth_mask']
        indicating_mask = batch['observed_mask'] - cond_mask
        res = {
            'X_ori' : batch['observed_data'].permute(0, 2, 1),
            'X' : batch['observed_data'].permute(0, 2, 1),
            'indicating_mask' : indicating_mask.permute(0, 2, 1),
            'observed_tp' : batch['timepoints'],
            'cond_mask' : cond_mask.permute(0, 2, 1),
        }
        return res

    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        inputs = self._get_input(batch, training)
        res = self.model.forward(inputs, training)
        return res['loss']
    
    def impute(self, batch:dict) -> torch.Tensor:
        inputs = self._get_input(batch, False)
        res = self.model.forward(inputs, False, self.n_samples)
        return res['imputed_data']


class _CSDI(nn.Module):
    def __init__(
        self,
        n_features,
        n_layers,
        n_heads,
        n_channels,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_time_embedding = d_time_embedding
        self.is_unconditional = is_unconditional

        self.embed_layer = nn.Embedding(
            num_embeddings=n_features,
            embedding_dim=d_feature_embedding,
        )
        self.backbone = BackboneCSDI(
            n_layers,
            n_heads,
            n_channels,
            n_features,
            d_time_embedding,
            d_feature_embedding,
            d_diffusion_embedding,
            is_unconditional,
            n_diffusion_steps,
            schedule,
            beta_start,
            beta_end,
        )

    @staticmethod
    def time_embedding(pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=pos.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = observed_tp.device
        time_embed = self.time_embedding(
            observed_tp, self.d_time_embedding
        )  # (B,L,emb)
        time_embed = time_embed.to(device)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.n_features).to(device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat(
            [time_embed, feature_embed], dim=-1
        )  # (B,L,K,emb+d_feature_embedding)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, inputs, training, n_sampling_times=0):
        results = {}
        if training:  # for training
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X_ori"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask)
            training_loss = self.backbone.calc_loss(
                observed_data, cond_mask, indicating_mask, side_info, training
            )
            results["loss"] = training_loss
        elif not training and n_sampling_times == 0:  # for validating
            (observed_data, indicating_mask, cond_mask, observed_tp) = (
                inputs["X"],
                inputs["indicating_mask"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask)
            validating_loss = self.backbone.calc_loss_valid(
                observed_data, cond_mask, indicating_mask, side_info, training
            )
            results["loss"] = validating_loss
        elif not training and n_sampling_times > 0:  # for testing
            observed_data, cond_mask, observed_tp = (
                inputs["X"],
                inputs["cond_mask"],
                inputs["observed_tp"],
            )
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.backbone(
                observed_data, cond_mask, side_info, n_sampling_times
            )  # (n_samples, n_sampling_times, n_features, n_steps)
            repeated_obs = observed_data.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            repeated_mask = cond_mask.unsqueeze(1).repeat(1, n_sampling_times, 1, 1)
            imputed_data = repeated_obs + samples * (1 - repeated_mask)

            results["imputed_data"] = imputed_data.permute(
                0, 1, 3, 2
            )  # (n_samples, n_sampling_times, n_steps, n_features)

        return results


class BackboneCSDI(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        n_channels,
        d_target,
        d_time_embedding,
        d_feature_embedding,
        d_diffusion_embedding,
        is_unconditional,
        n_diffusion_steps,
        schedule,
        beta_start,
        beta_end,
    ):
        super().__init__()

        self.d_target = d_target
        self.d_time_embedding = d_time_embedding
        self.d_feature_embedding = d_feature_embedding
        self.is_unconditional = is_unconditional
        self.n_channels = n_channels
        self.n_diffusion_steps = n_diffusion_steps

        d_side = d_time_embedding + d_feature_embedding
        if self.is_unconditional:
            d_input = 1
        else:
            d_side += 1  # for conditional mask
            d_input = 2

        self.diff_model = CsdiDiffusionModel(
            n_diffusion_steps,
            d_diffusion_embedding,
            d_input,
            d_side,
            n_channels,
            n_heads,
            n_layers,
        )

        # parameters for diffusion models
        if schedule == "quad":
            self.beta = (
                np.linspace(beta_start**0.5, beta_end**0.5, self.n_diffusion_steps)
                ** 2
            )
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, self.n_diffusion_steps)
        else:
            raise ValueError(
                f"The argument schedule should be 'quad' or 'linear', but got {schedule}"
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.register_buffer(
            "alpha_torch", torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        )

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def calc_loss_valid(
        self, observed_data, cond_mask, indicating_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.n_diffusion_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.n_diffusion_steps

    def calc_loss(
        self, observed_data, cond_mask, indicating_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        device = observed_data.device
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(device)
        else:
            t = torch.randint(0, self.n_diffusion_steps, [B]).to(device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diff_model(total_input, side_info, t)  # (B,K,L)

        target_mask = indicating_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def forward(self, observed_data, cond_mask, side_info, n_sampling_times):
        B, K, L = observed_data.shape
        device = observed_data.device
        imputed_samples = torch.zeros(B, n_sampling_times, K, L).to(device)

        for i in range(n_sampling_times):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.n_diffusion_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.n_diffusion_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diff_model(
                    diff_input, side_info, torch.tensor([t]).to(device)
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples


class CsdiDiffusionModel(nn.Module):
    def __init__(
        self,
        n_diffusion_steps,
        d_diffusion_embedding,
        d_input,
        d_side,
        n_channels,
        n_heads,
        n_layers,
    ):
        super().__init__()
        self.diffusion_embedding = CsdiDiffusionEmbedding(
            n_diffusion_steps=n_diffusion_steps,
            d_embedding=d_diffusion_embedding,
        )
        self.input_projection = conv1d_with_init(d_input, n_channels, 1)
        self.output_projection1 = conv1d_with_init(n_channels, n_channels, 1)
        self.output_projection2 = conv1d_with_init(n_channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                CsdiResidualBlock(
                    d_side=d_side,
                    n_channels=n_channels,
                    diffusion_embedding_dim=d_diffusion_embedding,
                    nheads=n_heads,
                )
                for _ in range(n_layers)
            ]
        )
        self.n_channels = n_channels

    def forward(self, x, cond_info, diffusion_step):
        (
            n_samples,
            input_dim,
            n_features,
            n_steps,
        ) = x.shape  # n_samples, 2, n_features, n_steps

        x = x.reshape(n_samples, input_dim, n_features * n_steps)
        x = self.input_projection(x)  # n_samples, n_channels, n_features*n_steps
        x = F.relu(x)
        x = x.reshape(n_samples, self.n_channels, n_features, n_steps)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(n_samples, self.n_channels, n_features * n_steps)
        x = self.output_projection1(x)  # (n_samples, channel, n_features*n_steps)
        x = F.relu(x)
        x = self.output_projection2(x)  # (n_samples, 1, n_features*n_steps)
        x = x.reshape(n_samples, n_features, n_steps)
        return x


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class CsdiDiffusionEmbedding(nn.Module):
    def __init__(self, n_diffusion_steps, d_embedding=128, d_projection=None):
        super().__init__()
        if d_projection is None:
            d_projection = d_embedding
        self.register_buffer(
            "embedding",
            self._build_embedding(n_diffusion_steps, d_embedding // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(d_embedding, d_projection)
        self.projection2 = nn.Linear(d_projection, d_projection)

    @staticmethod
    def _build_embedding(n_steps, d_embedding=64):
        steps = torch.arange(n_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (
            torch.arange(d_embedding) / (d_embedding - 1) * 4.0
        ).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

    def forward(self, diffusion_step: int):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class CsdiResidualBlock(nn.Module):
    def __init__(self, d_side, n_channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, n_channels)
        self.cond_projection = conv1d_with_init(d_side, 2 * n_channels, 1)
        self.mid_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)
        self.output_projection = conv1d_with_init(n_channels, 2 * n_channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=n_channels)
        self.feature_layer = get_torch_trans(
            heads=nheads, layers=1, channels=n_channels
        )

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape  # bz, 2, n_features, n_steps
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape  # bz, 2, n_features, n_steps
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)

        y = x + diffusion_emb
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
