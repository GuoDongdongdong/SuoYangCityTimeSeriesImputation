import torch
import numpy as np
from scipy.interpolate import interp1d

from utils.ExperimentArgs import ExperimentArgs
from models.BaseImpuateModel import BaseImputeModel
from utils.datasets import SuoYangCityDataset


class Model(BaseImputeModel):
    def __init__(self, exp_args:ExperimentArgs):
        super().__init__()
        self.exp_args = exp_args
        self.targets = exp_args['targets']
        self.checkpoints_path = exp_args.get_save_path()
        self.kind = exp_args['kind']

    def evaluate(self, batch:dict, training:bool) -> torch.Tensor:
        raise NotImplementedError('Statistical Model do not need to trian!')

    def impute(self, batch:dict) -> torch.Tensor:
        raise NotImplementedError('Statistical Model should use themself test function to impute!')
    
    def test(self, test_dataset:SuoYangCityDataset) -> None:
        imputed_data = test_dataset.unnorm_observed_data.copy()
        imputed_data[test_dataset.ground_truth_mask == 0] = np.nan
        L, D = imputed_data.shape
        for dim in range(D):
            data = imputed_data[:, dim]
            nan_idx = np.argwhere(~np.isnan(data)).squeeze()
            x = np.arange(1, L + 1)
            x = x[nan_idx]
            data_without_nan = data[nan_idx]
            interpolate_func = interp1d(x, data_without_nan, kind=self.kind, axis=0)
            axis_y = np.linspace(1, L, L)
            imputed_data[:, dim] = interpolate_func(axis_y)
        imputed_data = np.expand_dims(imputed_data, axis=0)
        test_dataset.save_result(imputed_data)