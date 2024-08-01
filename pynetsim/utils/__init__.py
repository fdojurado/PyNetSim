import random
import numpy as np
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


class RandomNumberGenerator:
    """
    Random number generator.

    :param seed: Seed for the random number generator
    :type seed: int
    """

    def __init__(self, config):
        self.seed = config.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.info("Random seed: %s", self.seed)

    def get_random(self):
        """Generate a random float between 0 and 1."""
        return random.random()

    def get_random_int(self, low, high):
        """Generate a random integer between low (inclusive) and high (inclusive)."""
        return random.randint(low, high)

    def get_random_choice(self, seq):
        """Return a random element from the non-empty sequence seq."""
        return random.choice(seq)

    def get_np_random(self):
        """Generate a random float between 0 and 1 using NumPy."""
        return np.random.random()

    def get_np_random_int(self, low, high):
        """Generate a random integer between low (inclusive) and high (inclusive) using NumPy."""
        return np.random.randint(low, high + 1)

    def get_np_random_choice(self, seq, size=None, replace=True):
        """
        Return a random sample from a given 1-D array using NumPy.

        :param seq: 1-D array-like or list from which to sample.
        :type seq: list or np.ndarray
        :param size: Number of samples to draw.
        :type size: int or None
        :param replace: Whether the sample is with replacement.
        :type replace: bool
        :return: Random sample from seq.
        :rtype: np.ndarray
        """
        return np.random.choice(seq, size=size, replace=replace)

    def get_uniform(self, a, b):
        """
        Generate a random float between a and b.

        :param a: Lower bound of the range.
        :type a: float
        :param b: Upper bound of the range.
        :type b: float
        :return: Random float between a and b.
        :rtype: float
        """
        return random.uniform(a, b)
