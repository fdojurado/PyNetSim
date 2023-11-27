from rich.logging import RichHandler

import logging
import logging.config
import time


class PyNetSimLogger:
    def __init__(self, log_file, namespace=None, stream_handler_level=logging.INFO,
                 file_handler_level=logging.DEBUG):
        self.log_file = log_file
        self.logger = logging.getLogger(namespace if namespace else "PyNetSim")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(message)s')
        # self.logger.setLevel(logging.DEBUG)

        # Create a stream handler
        stream_handler = RichHandler(rich_tracebacks=True)
        stream_handler.setLevel(stream_handler_level)
        stream_handler.setFormatter(formatter)

        # Create a file handler
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s |  %(levelname)s: %(message)s')
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_file, when='midnight', backupCount=30)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(file_handler_level)

        # Add handlers to the logger
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def get_logger(self):
        return self.logger


logger_utility = PyNetSimLogger(namespace=__name__, log_file="my_log.log")
logger = logger_utility.get_logger()


class Timer:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        if self.name:
            logger.info(f"{self.name}: {self.end - self.start}")
        else:
            logger.info(f"Time taken: {self.end - self.start}")
