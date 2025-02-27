'''
    Chapter 3 Ablation Experiment 2
    impute all nan in four datasets, and use MPformer forecast them.
'''

import os
import configparser
import subprocess
from configparser import ConfigParser

CONFIG_FILE_NAME = 'config.ini'
CONFIG_FILE_DIR  = '.'
TEMP_CONFIG_FILE_NAME = 'temp_config.ini'
TEMP_FILE_DIR = './temp'
IMPUTE_DIR = 'chapter3_experiment1_log'
CHECKPOINTS_fILE_NAME = 'checkpoints.pth'

DATASETS = [
    ('humidity_30per_block_missing.csv', 'humidity_missing'),
    ('temperature_30per_block_missing.csv', 'temperature_missing'),
    ('windspeed_30per_block_missing.csv', 'windspeed_missing'),
    ('water_30per_block_missing.csv', 'water_missing'),
]

MODEL_LIST = [
    'BRITS',
    'CSDI',
    'GRUD',
    'SAITS',
    'TIEGAN',
    'USGAN'
]

def run():
    args = ['-config_file_dir',
            TEMP_FILE_DIR, 
            '-config_file_name', 
            TEMP_CONFIG_FILE_NAME,
            ]
    result = subprocess.run(['python', 'run.py'] + args)

def shutdown():
    os.system("/usr/bin/shutdown")

def common_args_define(config:ConfigParser):
    config['CommonArgs']['dataset_file_dir'] = 'str:dataset'
    config['CommonArgs']['dataset_file_name'] = 'str:TODO'
    config['CommonArgs']['model'] = 'str:TODO'
    config['CommonArgs']['train_test'] = 'bool:False'
    config['CommonArgs']['targets'] = 'list:TODO'
    config['CommonArgs']['date_frequence'] = 'str:s'
    config['CommonArgs']['lookback_length'] = 'int:48'
    config['CommonArgs']['train_ratio'] = 'float:0.0'
    config['CommonArgs']['vali_ratio'] = 'float:0.0'
    config['CommonArgs']['artifical_missing_ratio'] = 'float:0.1'
    config['CommonArgs']['artifical_missing_type'] = 'str:block_missing'
    config['CommonArgs']['random_seed'] = 'int:202221543'
    config['CommonArgs']['batch_size'] = 'int:32'
    config['CommonArgs']['lr'] = 'float:1e-3'
    config['CommonArgs']['epochs'] = 'int:300'
    config['CommonArgs']['patience'] = 'int:5'
    config['CommonArgs']['num_workers'] = 'int:8'

if __name__ == '__main__':
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    for model_name in os.listdir(IMPUTE_DIR):
        if model_name not in MODEL_LIST:
            continue
        model_path = os.path.join(IMPUTE_DIR, model_name)
        for i, date in enumerate(os.listdir(model_path)):
            model_args_path = os.path.join(model_path, CHECKPOINTS_fILE_NAME)
            common_args_define(config)
            config['CommonArgs']['model_save_path'] = f'str:{model_args_path}'
            config['CommonArgs']['model'] = f'str:{model_name}'
            config['CommonArgs']['dataset_file_name'] = f'str:{DATASETS[i][0]}'
            config['CommonArgs']['targets'] = f'list:{DATASETS[i][1]}'
            with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
                config.write(f)
            run()
    shutdown()