'''
    Chapter 3 Experiment 1
'''

import os
import configparser
import subprocess

CONFIG_FILE_NAME = 'config.ini'
CONFIG_FILE_DIR  = '.'
TEMP_CONFIG_FILE_NAME = 'temp_config.ini'
TEMP_FILE_DIR = './temp'

MODEL_LIST = [
    'Interpolate',
    'USGAN',
    'CSDI',
    'BRITS',
    'GRUD',
    'SAITS',
    'TIEGAN',
]
INTERPOLATE = [
    'previous', # LOCF
    'cubic',
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

if __name__ == '__main__':
    config_file_path = os.path.join(CONFIG_FILE_DIR, CONFIG_FILE_NAME)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    os.makedirs(TEMP_FILE_DIR, exist_ok=True)
    for model_name in MODEL_LIST:
        config['CommonArgs']['model'] = f'str:{model_name}'
        if model_name == 'Interpolate':
            for kind_name in INTERPOLATE:
                config['Interpolate']['kind'] = f'str:{kind_name}'
                with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
                    config.write(f)
                run()
        else:
            with open(os.path.join(TEMP_FILE_DIR, TEMP_CONFIG_FILE_NAME), 'w') as f:
                config.write(f)
            run()
    shutdown()