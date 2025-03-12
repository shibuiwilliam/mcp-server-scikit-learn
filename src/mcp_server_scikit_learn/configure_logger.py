import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", logging.DEBUG)  # type: ignore
if isinstance(LOG_LEVEL, str):
    LOG_LEVEL = LOG_LEVEL.upper()


def make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d] [%(funcName)s] %(message)s"
    )
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(LOG_LEVEL)
    logger.addHandler(handler)

    return logger
