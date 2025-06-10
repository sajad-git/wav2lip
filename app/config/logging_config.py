"""
Logging Configuration for Avatar Streaming Service
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "/app/logs"):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general application logs
    app_log_file = os.path.join(log_dir, "avatar_service.log")
    file_handler = logging.handlers.RotatingFileHandler(
        app_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error log file for errors and above
    error_log_file = os.path.join(log_dir, "errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance log for processing metrics
    performance_log_file = os.path.join(log_dir, "performance.log")
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=3,
        encoding='utf-8'
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(detailed_formatter)
    
    # Create performance logger
    performance_logger = logging.getLogger("performance")
    performance_logger.addHandler(performance_handler)
    performance_logger.setLevel(logging.INFO)
    performance_logger.propagate = False
    
    # Avatar operations log
    avatar_log_file = os.path.join(log_dir, "avatar_operations.log")
    avatar_handler = logging.handlers.RotatingFileHandler(
        avatar_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=3,
        encoding='utf-8'
    )
    avatar_handler.setLevel(logging.INFO)
    avatar_handler.setFormatter(detailed_formatter)
    
    # Create avatar logger
    avatar_logger = logging.getLogger("avatar")
    avatar_logger.addHandler(avatar_handler)
    avatar_logger.setLevel(logging.INFO)
    avatar_logger.propagate = False
    
    # WebSocket log for connection tracking
    websocket_log_file = os.path.join(log_dir, "websocket.log")
    websocket_handler = logging.handlers.RotatingFileHandler(
        websocket_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=2,
        encoding='utf-8'
    )
    websocket_handler.setLevel(logging.INFO)
    websocket_handler.setFormatter(detailed_formatter)
    
    # Create websocket logger
    websocket_logger = logging.getLogger("websocket")
    websocket_logger.addHandler(websocket_handler)
    websocket_logger.setLevel(logging.INFO)
    websocket_logger.propagate = False
    
    # Suppress verbose logs from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("insightface").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Log setup completion
    logging.info(f"🔍 Logging configured - Level: {log_level}, Directory: {log_dir}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific component"""
    return logging.getLogger(name)


def log_performance_metric(operation: str, duration: float, additional_data: dict = None):
    """Log performance metrics in structured format"""
    logger = logging.getLogger("performance")
    
    metric_data = {
        "operation": operation,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_data:
        metric_data.update(additional_data)
    
    logger.info(f"PERFORMANCE: {metric_data}")


def log_avatar_operation(operation: str, avatar_id: str, user_id: str = None, 
                        status: str = "success", additional_data: dict = None):
    """Log avatar operations in structured format"""
    logger = logging.getLogger("avatar")
    
    operation_data = {
        "operation": operation,
        "avatar_id": avatar_id,
        "user_id": user_id,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_data:
        operation_data.update(additional_data)
    
    logger.info(f"AVATAR_OP: {operation_data}")


def log_websocket_event(event: str, client_id: str = None, additional_data: dict = None):
    """Log WebSocket events in structured format"""
    logger = logging.getLogger("websocket")
    
    event_data = {
        "event": event,
        "client_id": client_id,
        "timestamp": datetime.now().isoformat()
    }
    
    if additional_data:
        event_data.update(additional_data)
    
    logger.info(f"WEBSOCKET: {event_data}")


class StructuredLogger:
    """Structured logger for consistent logging format"""
    
    def __init__(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)
    
    def log_structured(self, level: str, category: str, message: str, **kwargs):
        """Log structured message with additional data"""
        log_data = {
            "category": category,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        log_data.update(kwargs)
        
        log_method = getattr(self.logger, level.lower())
        log_method(f"{category.upper()}: {log_data}")
    
    def info(self, category: str, message: str, **kwargs):
        self.log_structured("info", category, message, **kwargs)
    
    def error(self, category: str, message: str, **kwargs):
        self.log_structured("error", category, message, **kwargs)
    
    def warning(self, category: str, message: str, **kwargs):
        self.log_structured("warning", category, message, **kwargs)
    
    def debug(self, category: str, message: str, **kwargs):
        self.log_structured("debug", category, message, **kwargs) 