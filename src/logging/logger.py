"""Unified logging configuration for the RAG system."""

import logging
import os
from pathlib import Path
from typing import Optional

from src.config import load_config


_INITIALIZED = False


def setup_logging(config: Optional[dict] = None) -> None:
    """Initialize logging with settings from config.

    Args:
        config: Optional config dict. If None, loads from settings.yaml.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    if config is None:
        config = load_config()

    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("log_dir", "./logs")
    log_level = log_cfg.get("level", "INFO").upper()
    log_format = log_cfg.get(
        "log_format",
        "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("rag")
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level, logging.INFO))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler - app.log
    app_log_path = os.path.join(log_dir, "app.log")
    file_handler = logging.FileHandler(app_log_path, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level, logging.INFO))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Ingestion file handler
    ingestion_log_path = os.path.join(log_dir, "ingestion.log")
    ingestion_handler = logging.FileHandler(ingestion_log_path, encoding="utf-8")
    ingestion_handler.setLevel(getattr(logging, log_level, logging.INFO))
    ingestion_handler.setFormatter(formatter)
    ingestion_logger = logging.getLogger("rag.ingestion")
    ingestion_logger.addHandler(ingestion_handler)

    # Evaluation file handler
    eval_log_path = os.path.join(log_dir, "evaluation.log")
    eval_handler = logging.FileHandler(eval_log_path, encoding="utf-8")
    eval_handler.setLevel(getattr(logging, log_level, logging.INFO))
    eval_handler.setFormatter(formatter)
    eval_logger = logging.getLogger("rag.evaluation")
    eval_logger.addHandler(eval_handler)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the 'rag' namespace.

    Args:
        name: Logger name suffix (e.g., 'ingestion.pdf_parser').

    Returns:
        Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(f"rag.{name}")
