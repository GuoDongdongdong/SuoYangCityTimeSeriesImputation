
'''
    python configparser do not distinguish upper lower case in key, so use lowercase in config file.
'''

import os
import argparse
import configparser
from typing import Any, Callable

import torch

from utils.config import DEFAULT_CONFIG_FILE_NAME, DEFAULT_SPLIT_SYMBOL, EXPERIMENT_TITLE, TIME_NOW, LOG_SAVE_DIR

class ExperimentArgs:
    def __init__(self) -> None:
        command_parse = argparse.ArgumentParser(EXPERIMENT_TITLE)
        command_parse.add_argument('-config_file_dir', type=str, default='.', help='where to find config file.')
        command_parse.add_argument('-config_file_name', type=str, default='config.ini')
        command_args = command_parse.parse_args()
        config_path = os.path.join(command_args.config_file_dir, command_args.config_file_name)
        self.raw_config = configparser.ConfigParser()
        self.raw_config.read(config_path)
        config  = self._parse_type(self.raw_config)
        model_name = config['CommonArgs']['model']
        self.args = dict()
        self.args.update(config['CommonArgs'])
        # Model maybe have not any args.
        assert model_name in self.raw_config.sections(), f'not found {model_name} section in {DEFAULT_CONFIG_FILE_NAME}'
        self.args.update(config[model_name])
        self._check_args()
    
    def __getitem__(self, index:str) -> Any:
        return self.args[index]

    def _parse_type(self, config:configparser.ConfigParser) -> dict:
        def _str2bool(val:str) -> bool:
            assert val == 'True' or val == 'False', 'bool type should be [True False]'
            return val == 'True'
        
        def _str2list(val:str) -> list:
            return val.split(DEFAULT_SPLIT_SYMBOL)
        
        def _str2tuple(val:str) -> tuple:
            return tuple(val.split(DEFAULT_SPLIT_SYMBOL))
        
        def _type_func(func_name:str) -> Callable:
            if func_name == 'bool':
                return _str2bool
            if func_name == 'list':
                return _str2list
            if func_name == 'tuple':
                return _str2tuple
            return eval(func_name)
        
        res = dict()
        for section in config.sections():
            section_val = dict()
            for key, val in config[section].items():
                t, v = str.split(val, ':')
                section_val[key] = _type_func(t)(v)
            res[section] = section_val
        return res

    def _check_args(self) -> None:
        if self.args['use_gpu']:
            assert torch.cuda.is_available(), 'use GPU but cuda is not available!'

    def get_save_path(self) -> str:
        model = self.args['model']
        if model == 'Interpolate':
            # model_kind to distinguish statistical model.
            kind = self.args['kind']
            return os.path.join(LOG_SAVE_DIR, f'{model}_{kind}', TIME_NOW)
        else:
            return os.path.join(LOG_SAVE_DIR, model, TIME_NOW)

    def save_args(self) -> None:
        log_path = self.get_save_path()
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, DEFAULT_CONFIG_FILE_NAME), 'w') as f:
            self.raw_config.write(f)
