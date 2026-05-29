"""
Structured logging configuration.
Production: JSON format (for log aggregation / monitoring).
Development: Pretty format (for human readability).
"""

import logging
import sys
from typing import Any

from src.config import settings


class JSONFormatter(logging.Formatter):
    """
    Simple JSON formatter for production logs.
    In a real production system you might use python-json-logger.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Include extra fields if present
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        if hasattr(record, "agent"):
            log_obj["agent"] = record.agent
            
        # Include exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"agent": "reviewer"})
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG if settings.langchain_tracing_v2 else logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    
    # Simple pretty format for dev, JSON for production
    # You can toggle this with an env var later if needed
    if settings.langchain_tracing_v2:
        # When tracing is on, we assume development mode
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
    else:
        # Production-style JSON
        formatter = JSONFormatter()
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger