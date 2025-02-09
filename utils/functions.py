import random
from typing import Optional, Union

import torch
import numpy as np


def fix_random_seed(random_seed:int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _check_inputs(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    check_shape: bool = True,
):
    # check type
    assert isinstance(predictions, type(targets)), (
        f"types of `predictions` and `targets` must match, but got"
        f"`predictions`: {type(predictions)}, `target`: {type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    # check shape
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert (
            prediction_shape == target_shape
        ), f"shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}"
    # check NaN
    assert not lib.isnan(
        predictions
    ).any(), "`predictions` mustn't contain NaN values, but detected NaN in it"
    assert not lib.isnan(
        targets
    ).any(), "`targets` mustn't contain NaN values, but detected NaN in it"

    if masks is not None:
        # check type
        assert isinstance(masks, type(targets)), (
            f"types of `masks`, `predictions`, and `targets` must match, but got"
            f"`masks`: {type(masks)}, `targets`: {type(targets)}"
        )
        # check shape, masks shape must match targets
        mask_shape = masks.shape
        assert mask_shape == target_shape, (
            f"shape of `masks` must match `targets` shape, "
            f"but got `mask`: {mask_shape} that is different from `targets`: {target_shape}"
        )
        # check NaN
        assert not lib.isnan(
            masks
        ).any(), "`masks` mustn't contain NaN values, but detected NaN in it"

    return lib

def calc_mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mse = calc_mse(predictions, targets)

    mse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`, so the result is 5/5=1.

    If we want to prevent some values from MSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mse = calc_mse(predictions, targets, masks)

    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.square(predictions - targets))

def calc_mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Absolute Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = calc_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = calc_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (
            lib.sum(masks) + 1e-12
        )
    else:
        return lib.mean(lib.abs(predictions - targets))

def calc_rmse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Root Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_rmse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> rmse = calc_rmse(predictions, targets)

    rmse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`,
    so the result is :math:`\\sqrt{5/5}=1`.

    If we want to prevent some values from RMSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> rmse = calc_rmse(predictions, targets, masks)

    rmse = 0.707 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # don't have to check types and NaN here, since calc_mse() will do it
    lib = np if isinstance(predictions, np.ndarray) else torch
    return lib.sqrt(calc_mse(predictions, targets, masks))

def get_deltas(mask:np.ndarray) -> np.ndarray:
    assert mask.ndim == 2, f"mask shape should like [length, dim], but got {mask.shape}!"
    def func(col):
        res = np.zeros_like(col, dtype=np.float32)
        for i in range(1, len(col)):
            if col[i] != 0:
                res[i] = 1.0
            else :
                res[i] = 1.0 + res[i - 1]
        return res
    deltas = np.apply_along_axis(func, 0, mask)
    return deltas

def get_ground_truth_mask(data:np.ndarray, p:float, missing_type:str) -> np.ndarray:
    assert missing_type in ['mcar', 'block_missing', 'random_block_mix'], f'missing type only support MCAR, bug got {missing_type}!'
    assert 0 < p < 1, f'missing rate should in range (0, 1), but got {p}!'
    if missing_type == 'mcar':
        return mcar(data, p)
    elif missing_type == 'block_missing':
        return block_missing(data, p)
    elif missing_type == 'random_block_mix':
        return random_block_mix(data, p)

def mcar(data:np.ndarray, p:float) -> np.ndarray:
    '''
        missing complete at random.
        set some observed data to nan with probability p.
    ''' 
    observed_mask = (1 - np.isnan(data)).reshape(-1)
    observed_index = np.where(observed_mask)[0]
    artifical_missing_index = np.random.choice(observed_index,
                                               int(len(observed_index) * p),
                                               replace=False)
    observed_mask[artifical_missing_index] = False
    observed_mask = observed_mask.reshape(data.shape)
    return observed_mask

def block_missing(data:np.ndarray, p:float) -> np.ndarray:
    observed_mask = (1 - np.isnan(data)).reshape(-1)
    observed_index = np.where(observed_mask)[0]
    missing_number = int(sum(observed_mask) * p)
    patch = 24
    missing_points = int(missing_number / patch)
    artifical_missing_index = np.random.choice(observed_index, missing_points, replace=False)
    for idx in artifical_missing_index:
        observed_mask[idx : min(idx + patch, len(observed_index))] = False
    observed_mask = observed_mask.reshape(data.shape)
    return observed_mask

def random_block_mix(data:np.ndarray, p:float) -> np.ndarray:
    '''
        we use p * 0.5 rate to block mask and p * 0.5 to random mask. 
    '''
    observed_mask = (1 - np.isnan(data)).reshape(-1)
    observed_index = np.where(observed_mask)[0]
    missing_number = int(sum(observed_mask) * p)
    block_missing_number = int(missing_number * 0.5)
    random_missing_number = missing_number - block_missing_number
    patch = 24
    missing_points = int(block_missing_number / patch)
    artifical_missing_index = np.random.choice(observed_index, missing_points, replace=False)
    for idx in artifical_missing_index:
        observed_mask[idx : min(idx + patch, len(observed_index))] = False
    observed_index = np.where(observed_mask)[0]
    artifical_missing_index = np.random.choice(observed_index, random_missing_number, replace=False)
    observed_mask[artifical_missing_index] = False
    observed_mask = observed_mask.reshape(data.shape)
    return observed_mask


def locf(X:torch.Tensor|np.ndarray, first_step_imputation:str="backward") -> torch.Tensor|np.ndarray:
    assert X.ndim == 3, 'X shape should be like [B, L, D]'
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        return locf_torch(X, first_step_imputation).numpy()
    return locf_torch(X, first_step_imputation)

def locf_torch(X:torch.Tensor, first_step_imputation:str="backward") -> torch.Tensor:
    """Torch implementation of LOCF.

    Parameters
    ----------
    X : tensor,
        Time series containing missing values (NaN) to be imputed.

    first_step_imputation : str, default='backward'
        With LOCF, the observed values are carried forward to impute the missing ones. But if the first value
        is missing, there is no value to carry forward. This parameter is used to determine the strategy to
        impute the missing values at the beginning of the time-series sequence after LOCF is applied.
        It can be one of ['backward', 'zero', 'median', 'nan'].
        If 'nan', the missing values at the sequence beginning will be left as NaNs.
        If 'zero', the missing values at the sequence beginning will be imputed with 0.
        If 'backward', the missing values at the beginning of the time-series sequence will be imputed with the
        first observed value in the sequence, i.e. the first observed value will be carried backward to impute
        the missing values at the beginning of the sequence. This method is also known as NOCB (Next Observation
        Carried Backward). If 'median', the missing values at the sequence beginning will be imputed with the overall
        median values of features in the dataset.
        If `first_step_imputation` is not "nan", if missing values still exist (this is usually caused by whole feature
        missing) after applying `first_step_imputation`, they will be filled with 0.

    Returns
    -------
    X_imputed : tensor,
        Imputed time series.
    """
    trans_X = X.permute((0, 2, 1))
    mask = torch.isnan(trans_X)
    n_samples, n_steps, n_features = mask.shape
    idx = torch.where(~mask, torch.arange(n_features, device=mask.device), 0)
    idx = np.maximum.accumulate(idx, axis=2)

    collector = []
    for x, i in zip(trans_X, idx):
        collector.append(x[torch.arange(n_steps)[:, None], i])
    X_imputed = torch.stack(collector)
    X_imputed = X_imputed.permute((0, 2, 1))

    # If there are values still missing, they are missing at the beginning of the time-series sequence.
    if torch.isnan(X_imputed).any():
        if first_step_imputation == "nan":
            pass
        elif first_step_imputation == "zero":
            X_imputed = torch.nan_to_num(X_imputed, nan=0)
        elif first_step_imputation == "backward":
            # imputed by last observation carried backward (LOCB)
            X_imputed_transpose = X_imputed.clone()
            X_imputed_transpose = torch.flip(X_imputed_transpose, dims=[1])
            X_LOCB = locf_torch(
                X_imputed_transpose,
                "zero",
            )
            X_imputed = torch.flip(X_LOCB, dims=[1])
        elif first_step_imputation == "median":
            bz, n_steps, n_features = X_imputed.shape
            X_imputed_reshaped = X_imputed.clone().reshape(-1, n_features)
            median_values = torch.nanmedian(X_imputed_reshaped, dim=0)
            for i, v in enumerate(median_values.values):
                X_imputed[:, :, i] = torch.nan_to_num(X_imputed[:, :, i], nan=v)
            if torch.isnan(X_imputed).any():
                X_imputed = torch.nan_to_num(X_imputed, nan=0)

    return X_imputed


def calc_smape(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor], masks: Optional[Union[np.ndarray, torch.Tensor]]=None) -> Union[float, torch.Tensor]:
    lib = _check_inputs(predictions, targets, masks)
    epsilon = 1e-10
    denominator = (lib.abs(predictions) + lib.abs(targets)) / 2 + epsilon
    diff = lib.abs(predictions - targets) / denominator
    if masks is not None:
        diff = diff * masks
        return 100 * np.sum(diff) / np.sum(masks)
    return 100 * np.mean(diff)