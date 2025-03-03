"""Logging configuration for the text generation project."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


try:
    from dotenv import load_dotenv

    dotenv_path = Path(__file__).parent.parent.parent / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
except ImportError:
    pass


def setup_logger(
    name: str = "text_generation",
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Set up and configure a logger with appropriate formatting and handlers.

    :param name: Name of the logger
    :param level: Logging level (defaults to environment variable or INFO)
    :param log_file: Optional path to a log file
    :return: Configured logger instance
    """

    env_level = os.environ.get("TEXT_GEN_LOG_LEVEL", "INFO").upper()
    log_level = (
        level if level is not None else getattr(logging, env_level, logging.INFO)
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


train_logger = setup_logger("text_generation.train")
model_logger = setup_logger("text_generation.model")
generate_logger = setup_logger("text_generation.generate")
