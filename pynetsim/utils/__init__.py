from rich.logging import RichHandler

import logging
import logging.config

# Logger class


class PyNetSimLogger:
    def __init__(self, log_file, namespace=None):
        self.log_file = log_file
        self.logger = logging.getLogger(namespace if namespace else "PyNetSim")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(message)s')
        self.logger.setLevel(logging.DEBUG)

        # Create a stream handler
        stream_handler = RichHandler(rich_tracebacks=True)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        # Create a file handler
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s |  %(levelname)s: %(message)s')
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=self.log_file, when='midnight', backupCount=30)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

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
