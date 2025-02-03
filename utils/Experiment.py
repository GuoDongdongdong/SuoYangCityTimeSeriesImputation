import os

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer

from utils.EarlyStopping import EarlyStopping
from utils.ExperimentArgs import ExperimentArgs
from utils.logger import logger
from utils.config import GAN_MODEL_LIST, STATSTICAL_MODEL_LIST
from utils.datasets import data_provider
from models.BaseImpuateModel import BaseImputeModel
from models import Interpolate
from models import SAITS, ImputeFormer, TIEGAN
from models import BRITS, GRUD, MRNN
from models import TimesNet
from models import CSDI, GPVAE
from models import USGAN

MODELS = {
    'Interpolate'  : Interpolate,
    'SAITS'        : SAITS,
    'ImputeFormer' : ImputeFormer,
    'BRITS'        : BRITS,
    'GRUD'         : GRUD,
    'MRNN'         : MRNN,
    'TimesNet'     : TimesNet,
    'USGAN'        : USGAN,
    'CSDI'         : CSDI,
    'GPVAE'        : GPVAE,
    'TIEGAN'       : TIEGAN,
}


class Experiment:
    def __init__(self, exp_args:ExperimentArgs):
        self.exp_args = exp_args
        self.device = self._get_device()
        self.model = self._build_model()
        logger.info(f'Model parameter is {self.model.get_model_params()}') 

    def _get_device(self) -> torch.device:
        if self.exp_args['use_gpu']:
            GPU_id = self.exp_args['gpu_id']
            os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
            device = torch.device(f"cuda:{GPU_id}")
        else:
            device = torch.device('cpu')
        logger.debug(device)
        return device

    def _get_optimizer(self, model:nn.Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.exp_args['lr'])

    def _build_model(self) -> BaseImputeModel:
        model:BaseImputeModel = MODELS[self.exp_args['model']].Model(self.exp_args)
        if self.exp_args['use_multi_gpu']:
            gpu_id = self.exp_args['gpu_id']
            device_ids = list(gpu_id.split(','))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        inner_model = model.get_inner_model()
        if isinstance(inner_model, torch.nn.Module) :
            inner_model = inner_model.float().to(self.device)
        return model
    
    def _send_data_to_device(self, obj:dict) -> dict:
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, dtype=torch.float32)
        elif isinstance(obj, dict):
            return {key : self._send_data_to_device(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return list(self._send_data_to_device(item) for item in obj)
        elif isinstance(obj, tuple):
            return tuple(self._send_data_to_device(item) for item in obj)
        else:
            return obj 

    def load_model(self, save_path:str) -> None:
        self.model.load_model(save_path)

    def get_model_params(self) -> int:
        return self.model.get_model_params()

    def train_GAN(self):
        dataset, dataloader = data_provider(self.exp_args,'train')
        early_stop = EarlyStopping(
            self.model.get_inner_model(),
            self.exp_args['patience'],
            self.exp_args.get_save_path()
        )
        G_optimizer, D_optimizer = self._get_optimizer(self.model.get_generator()), self._get_optimizer(self.model.get_discriminator())
        if self.exp_args['use_amp'] :
            scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for epoch in range(self.exp_args['epochs']):
            self.model.get_inner_model().train()
            G_train_loss = []
            D_train_loss = []
            for idx, batch in enumerate(dataloader):
                batch = self._send_data_to_device(batch)
                if idx % self.exp_args['g_steps'] == 0:
                    G_optimizer.zero_grad()
                    loss = self.model.evaluate(batch, True, "generator")
                    if self.exp_args['use_amp']:
                        scaler.scale(loss).backward()
                        scaler.step(G_optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        G_optimizer.step()
                    G_train_loss.append(loss.item())
                if idx % self.exp_args['d_steps'] == 0:
                    D_optimizer.zero_grad()
                    loss = self.model.evaluate(batch, True, "discriminator")
                    if self.exp_args['use_amp']:
                        scaler.scale(loss).backward()
                        scaler.step(D_optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        D_optimizer.step()
                    D_train_loss.append(loss.item())
                    
            G_train_loss = np.average(G_train_loss)
            D_train_loss = np.average(D_train_loss)
            validate_loss = self._vali()
            logger.info("Epoch: {0} G_Train Loss: {1:.7f} D_Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(epoch + 1, G_train_loss, D_train_loss, validate_loss))

            early_stop.update(validate_loss)
            if early_stop.stop:
                logger.info("Early stopping")
                break

    def train(self) -> None:
        # statistical model jump trian process.
        if self.exp_args['model'] in STATSTICAL_MODEL_LIST:
            return
        # GAN model need use train_GAN
        if self.exp_args['model'] in GAN_MODEL_LIST:
            self.train_GAN()
            return
        dataset, dataloader = data_provider(self.exp_args, 'train')
        early_stop = EarlyStopping(
            self.model.get_inner_model(),
            self.exp_args['patience'],
            self.exp_args.get_save_path()
        )
        optimizer = self._get_optimizer(self.model.get_inner_model())
        if self.exp_args['use_amp'] :
            scaler = torch.cuda.amp.grad_scaler.GradScaler()
        for epoch in range(self.exp_args['epochs']):
            self.model.get_inner_model().train()
            train_loss_list = []
            for idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                batch = self._send_data_to_device(batch)
                train_loss = self.model.evaluate(batch, True)
                if self.exp_args['use_amp']:
                    scaler.scale(train_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
                    optimizer.step()
                train_loss_list.append(train_loss.item())
            train_loss = np.average(train_loss_list)
            validate_loss = self._vali()
            logger.info("epoch: {0} train loss: {1:.7f} validate Loss: {2:.7f}".format(epoch + 1, train_loss, validate_loss))
            early_stop.update(validate_loss)
            if early_stop.stop:
                logger.info("Early stopping")
                break
    
    def _vali(self) -> float:
        dataset, dataloader = data_provider(self.exp_args, 'validate')
        validation_loss_list = []
        self.model.get_inner_model().eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = self._send_data_to_device(batch)
                loss = self.model.evaluate(batch, False)
                validation_loss_list.append(loss.item())

        validation_loss = np.average(validation_loss_list)
        return validation_loss

    def test(self) -> None:
        dataset, dataloader = data_provider(self.exp_args, 'test')
        # statistical model need .
        if self.exp_args['model'] in STATSTICAL_MODEL_LIST:
            self.model.test(dataset)
            return
        with torch.no_grad():
            self.model.get_inner_model().eval()
            output_list = []
            for batch in dataloader:
                batch = self._send_data_to_device(batch)
                # [B, L, D] or [B, n_samples, L, D]
                output = self.model.impute(batch)
                output_list.append(output)
            output = torch.cat(output_list).cpu()
            output = dataset.inverse(output)
            dataset.save_result(output)
