'''
    Chapter 3 Ablation Experiment 1
'''

import os
import configparser
import subprocess
from configparser import ConfigParser

CONFIG_FILE_NAME = 'config.ini'
CONFIG_FILE_DIR  = '.'
TEMP_CONFIG_FILE_NAME = 'temp_config.ini'
TEMP_FILE_DIR = './temp'

DATASETS = [
    ('humidity_30per_block_missing.csv', 'humidity_missing'),
    ('temperature_30per_block_missing.csv', 'temperature_missing'),
    ('windspeed_30per_block_missing.csv', 'windspeed_missing'),
    ('water_30per_block_missing.csv', 'water_missing'),
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
    config['CommonArgs']['dataset_file_name'] = 'str:humidity_30per_block_missing.csv'
    config['CommonArgs']['model'] = 'str:TIEGAN'
    config['CommonArgs']['train_test'] = 'bool:True'
    config['CommonArgs']['targets'] = 'list:humidity_missing'
    config['CommonArgs']['date_frequence'] = 'str:s'
    config['CommonArgs']['lookback_length'] = 'int:48'
    config['CommonArgs']['train_ratio'] = 'float:0.7'
    config['CommonArgs']['vali_ratio'] = 'float:0.1'
    config['CommonArgs']['artifical_missing_ratio'] = 'float:0.1'
    config['CommonArgs']['artifical_missing_type'] = 'str:block_missing'
    config['CommonArgs']['random_seed'] = 'int:202221543'
    config['CommonArgs']['batch_size'] = 'int:32'
    config['CommonArgs']['lr'] = 'float:1e-3'
    config['CommonArgs']['epochs'] = 'int:300'
    config['CommonArgs']['patience'] = 'int:5'
    config['CommonArgs']['num_workers'] = 'int:8'

def row2(config:ConfigParser):
    #table 3.3 row 2
    common_args_define(config)
    config['CommonArgs']['artifical_missing_type'] = 'str:mcar'


def row3(config:ConfigParser):
    #table 3.3 row 3
    common_args_define(config)
    config['TIEGAN']['ort_weight'] = 'float:0.0'

def row4(config:ConfigParser):
    #table 3.3 row 4
    common_args_define(config)
    config['CommonArgs']['model'] = 'str:TIEGAN_wo_TIE'

def row5(config:ConfigParser):
    #table 3.3 row 5
    common_args_define(config)
    config['TIEGAN']['diagonal_attention_mask'] = 'bool:False'

def row6(config:ConfigParser):
    #table 3.3 row 6
    common_args_define(config)
    config['CommonArgs']['model'] = 'str:TIEGAN_wo_GAN'

if __name__ == '__main__':
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    exp_list = [row2, row3, row4, row5, row6]
    for dataset_name, targets_name in DATASETS:
        for exp in exp_list:
            exp(config)
            config['CommonArgs']['dataset_file_name'] = f'str:{dataset_name}'
            config['CommonArgs']['targets'] = f'list:{targets_name}'
            with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
                config.write(f)
            run()
    shutdown()