import os

import numpy as np
import pandas as pd

from utils.config import NAN_SYMBOL
from utils.functions import mcar, block_missing, random_block_mix


def mask_original_dataset(dir_name:str, file_name:str, missing_type:str, missing_rate: float, targets: list[str]) -> None:
    assert missing_type in ['mcar', 'block_missing', 'random_block_mix'], f'missing type should be mcar, block_missing, random_block_mix but got {missing_type}!'
    assert 0 < missing_rate < 1, f'missing rate should in range (0, 1), but got {missing_rate}!'
    file_path = os.path.join(dir_name, file_name)
    data = pd.read_csv(file_path)
    for col in targets:
        missing_col = col + '_missing'
        data[missing_col] = data[col]
        if missing_type == 'mcar':
            mask = mcar(np.array(data[col]), missing_rate)
        elif missing_type == 'block_missing':
            mask = block_missing(np.array(data[col]), missing_rate)
        elif missing_type == 'random_block_mix':
            mask = random_block_mix(np.array(data[col]), missing_rate)
        data[missing_col][mask == 0] = np.nan
    file_name, file_suffix = file_name.split('.')
    data.to_csv(f'{file_name}_{int(missing_rate * 100)}per_{missing_type}.{file_suffix}', index=False, na_rep=NAN_SYMBOL)
