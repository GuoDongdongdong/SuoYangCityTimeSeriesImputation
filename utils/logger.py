import os
import logging

from utils.config import DEFAULT_LOGGER_NAME, DEFAULT_LOGGER_FORMAT, DEFAULT_LOGGER_LEVEL, DEFAULT_LOGGER_DATE_FORMAT

LOGGER_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR
}

class LoggerWrapper:
    def __init__(self, 
                 logger_name:str=DEFAULT_LOGGER_NAME, 
                 logger_level:str=DEFAULT_LOGGER_LEVEL, 
                 logger_format:str=DEFAULT_LOGGER_FORMAT, 
                 logger_date_format:str=DEFAULT_LOGGER_DATE_FORMAT):
        assert logger_level in LOGGER_LEVEL.keys(), 'logger level shoulde one of [debug info warning error]'
        self.logger_level = logger_level
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        self.logger.setLevel(LOGGER_LEVEL[logger_level])
        stream = logging.StreamHandler()
        self.format = logging.Formatter(logger_format, logger_date_format)
        stream.setFormatter(self.format)
        self.logger.addHandler(stream)

    '''
        ouput log file to File
    '''
    def set_file_log(self, log_file_save_path:str) -> None:
        os.makedirs(log_file_save_path, exist_ok=True)
        file_handle = logging.FileHandler(os.path.join(log_file_save_path, DEFAULT_LOGGER_NAME))
        file_handle.setLevel(LOGGER_LEVEL[self.logger_level])
        file_handle.setFormatter(self.format)
        self.logger.addHandler(file_handle)

logger_wrapper = LoggerWrapper()
logger = logger_wrapper.logger