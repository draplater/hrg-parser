import logging

import sys

logFormatter = logging.Formatter(
    "%(asctime)s [%(levelname)-5.5s] [%(name)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logFormatter)
logger.addHandler(console_handler)


def log_to_file(path):
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(logFormatter)
    logger.addHandler(file_handler)
