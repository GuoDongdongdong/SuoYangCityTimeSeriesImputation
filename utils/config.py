'''
    this file describe all magic string.
'''

# ExperimentArgs configuration
import datetime


EXPERIMENT_TITLE         = 'SuoYang City Earthen Ruins Climate Imputation.'
DEFAULT_CONFIG_FILE_NAME = 'config.ini'
DEFAULT_CHECKPOINTS_NAME = 'checkpoints.pth'
DEFAULT_RESULT_FILE_NAME = 'result.csv'
STATSTICAL_MODEL_LIST    = ['Interpolate']
GAN_MODEL_LIST           = ['USGAN', 'TIEGAN']
NAN_SYMBOL               = 'NaN'

# Logger configuration
DEFAULT_LOGGER_NAME        = 'SuoYangCityLogger'
DEFAULT_LOGGER_LEVEL       = 'debug'
DEFAULT_LOGGER_FORMAT      = '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
DEFAULT_LOGGER_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOGGER_NAME        = 'logfile'
LOG_SAVE_DIR               = 'log'  # log information save to /LOG_SAVE_DIR/model_name/time_now/
TIME_NOW                   = datetime.datetime.now().__format__("%Y%m%d_T%H%M%S")
# datasets configuration 
DEFAULT_DATE_COLUMN_NAME = 'date'
DEFAULT_SPLIT_SYMBOL = ','