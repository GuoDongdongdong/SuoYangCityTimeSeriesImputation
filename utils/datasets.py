import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from utils.ExperimentArgs import ExperimentArgs
from utils.logger import logger
from utils.functions import get_ground_truth_mask, get_deltas, locf, calc_mae, calc_mse, calc_rmse
from utils.config import DEFAULT_DATE_COLUMN_NAME, DEFAULT_RESULT_FILE_NAME, NAN_SYMBOL
from utils.TimeFeature import time_features


'''
    Suo Yang City csv file column should be like `date` `targets` `other_column`
    data in csv file will be split into `train` `validate` `test` in order

'''
FLAG_DICT = {
    'train' : 0,
    'validate' : 1,
    'test' : 2
}
class SuoYangCityDataset(Dataset):
    '''
        self.raw_data is orignal data read from csv file.
        self.features is date + targets
        self.date is date column
        self.observed_data is normalized data by train dataset data and set Nan to zero.
        self.observed_mask is observed matrix.
        self.deltas is time gap matrix.
        note that we normalized data before set Nan to zero.
    '''
    def __init__(self, exp_args:ExperimentArgs, flag:str) -> None:
        super().__init__()
        assert flag in ['train', 'validate', 'test'], 'Dataset flag should be [train validate test]'
        self.exp_args = exp_args
        dataset_file_path = os.path.join(self.exp_args['dataset_file_dir'], self.exp_args['dataset_file_name'])
        self.raw_data = pd.read_csv(dataset_file_path)
        self.features = [DEFAULT_DATE_COLUMN_NAME] + self.exp_args['targets']
        self.raw_data = self.raw_data[self.features]
        data = self.raw_data[self.exp_args['targets']]
        self.record_length = len(self.raw_data)
        self.train_dataset_length = int(self.exp_args['train_ratio'] * self.record_length)
        self.validate_dataset_length = int(self.exp_args['vali_ratio'] * self.record_length)
        self.test_dataset_legth = self.record_length - self.train_dataset_length - self.validate_dataset_length
        self.board_l = [0, self.train_dataset_length, self.train_dataset_length + self.validate_dataset_length]
        self.board_r = [self.train_dataset_length, self.train_dataset_length + self.validate_dataset_length, self.record_length]
        self.date = self.raw_data[DEFAULT_DATE_COLUMN_NAME][self.board_l[FLAG_DICT[flag]] : 
        self.board_r[FLAG_DICT[flag]]]
        
        train_dataset = data[self.board_l[FLAG_DICT['train']] : self.board_r[FLAG_DICT['train']]]
        self.scaler = StandardScaler()
        self.scaler.fit(train_dataset)
        self.unnorm_observed_data:np.ndarray = data[self.board_l[FLAG_DICT[flag]] : self.board_r[FLAG_DICT[flag]]].values
        self.observed_data = self.scaler.transform(self.unnorm_observed_data)
        self.observed_mask = 1 - np.isnan(self.observed_data)
        self.ground_truth_mask = get_ground_truth_mask(self.observed_data, self.exp_args['artifical_missing_ratio'], self.exp_args['artifical_missing_type'])
        self.loss_mask = self.observed_mask - self.ground_truth_mask
        self.deltas = get_deltas(self.ground_truth_mask)
        self.observed_data = np.nan_to_num(self.observed_data)
        self.locf_data = locf(np.expand_dims(self.unnorm_observed_data, 0))
        self.locf_data = np.squeeze(self.locf_data, 0)
        self.empirical_mean = np.sum(self.unnorm_observed_data * self.observed_mask, axis=0) / np.sum(self.observed_mask, axis=0)
        self.empirical_mean = np.nan_to_num(self.empirical_mean, 0)

    def __getitem__(self, index:int) -> dict:
        l, r = index, index + self.exp_args['lookback_length']
        res = {
            'index'             : index,
            'observed_data'     : self.observed_data[l : r],
            'observed_mask'     : self.observed_mask[l : r],
            'ground_truth_mask' : self.ground_truth_mask[l : r],
            'deltas'            : self.deltas[l : r],
            'loss_mask'         : self.loss_mask[l : r],
            'timepoints'        : np.arange(self.exp_args['lookback_length']),
            'locf_data'         : self.locf_data[l : r],
            'empirical_mean'    : self.empirical_mean,
            'date'              : time_features(pd.to_datetime(self.date[l : r].values), self.exp_args['date_frequence']),
        }
        return res

    def __len__(self) -> int:
        return len(self.observed_data) - self.exp_args['lookback_length']

    def _inverse_ndarry(self, data:np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    def _inverse_tensor(self, data:torch.Tensor) -> torch.Tensor:
        res = self.scaler.inverse_transform(data)
        return torch.from_numpy(res)

    def inverse(self, data:torch.Tensor|np.ndarray) -> torch.Tensor | np.ndarray:
        data_shape = data.shape
        D = data_shape[-1]
        data = data.reshape(-1, D)
        if isinstance(data, torch.Tensor):
            data = self._inverse_tensor(data)
        elif isinstance(data, np.ndarray):
            data = self._inverse_ndarry(data)
        data = data.reshape(data_shape)
        return data

    def save_result(self, imputed_data:np.ndarray|torch.Tensor) -> None:
        # TODO: we have not impute all test dataset, need to fix it.
        assert imputed_data.ndim == 3 or imputed_data.ndim == 4, f'imputed_data shape should be like [B, L, D] or [B, n_smaples, L, D], but got{imputed_data.shape}!'
        if isinstance(imputed_data, torch.Tensor):
            imputed_data = imputed_data.numpy()
        if imputed_data.ndim == 4:
            # TODO we can use all n_samples's data to plot a probability range chart.
            imputed_data = np.median(imputed_data, axis=1)
        B, L, D = imputed_data.shape
        imputed_data_length = B * L
        imputed_data = imputed_data.reshape(imputed_data_length, D)
        observed_data = self.unnorm_observed_data.copy()
        observed_data = np.nan_to_num(observed_data)
        observed_data = observed_data[: imputed_data_length]
        target_mask = self.loss_mask.copy()
        target_mask = target_mask[: imputed_data_length]
        mae = calc_mae(imputed_data, observed_data, target_mask)
        mse = calc_mse(imputed_data, observed_data, target_mask)
        rmse = calc_rmse(imputed_data, observed_data, target_mask)
        logger.info(f"mae: {mae}")
        logger.info(f"mse : {mse}")
        logger.info(f"rmse : {rmse}")
        df = pd.DataFrame()
        padding_length = self.test_dataset_legth - imputed_data_length
        padding_data = np.full((padding_length, D), float('nan'))
        imputed_data = np.concatenate((imputed_data, padding_data))
        observed_data = self.unnorm_observed_data.copy()
        observed_data[self.ground_truth_mask == 0] = np.nan
        df['date'] = self.date
        df[self.exp_args['targets']] = observed_data
        df[[target + '_imputation' for target in self.exp_args['targets']]] = imputed_data
        save_path = os.path.join(self.exp_args.get_save_path(), DEFAULT_RESULT_FILE_NAME)
        df.to_csv(save_path, index=False, float_format='%.2f', na_rep=NAN_SYMBOL)

def data_provider(exp_args:ExperimentArgs, flag:str) -> tuple[SuoYangCityDataset, DataLoader]:
    assert flag in ['train', 'validate', 'test'], f'Dataset flag should be [train validate test], but got {flag}!'
    dataset = SuoYangCityDataset(exp_args, flag)
    shuffle = True if flag == 'train' else False
    sampler = None if flag == 'train' else iter(range(0, len(dataset), exp_args['lookback_length']))
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=exp_args['batch_size'], 
                            shuffle=shuffle, 
                            num_workers=exp_args['num_workers'], 
                            sampler=sampler
                            )
    return dataset, dataloader