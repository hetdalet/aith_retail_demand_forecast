import logging
import sys

def get_logger(level: str = logging.INFO,
               name: str = 'default') -> logging.Logger:
    logging.basicConfig(level=level)

    handler = logging.StreamHandler(sys.stdout)
    frmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(frmt)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger
