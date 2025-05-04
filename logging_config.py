import logging
import sys
import os
from datetime import datetime

def setup_logger(name='csm_adapter', level=logging.INFO, log_dir='logs'):
    \"\"\"Sets up a logger that writes to console and a timestamped file.\"\"\"
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Formatter
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    formatter = logging.Formatter(log_format)

    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Set root logger level if needed (e.g., for libraries)
    # logging.getLogger().setLevel(level)

    # Example: Quieten overly verbose libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    # Add others as needed

    return logger

# Example usage:
# from logging_config import setup_logger
# logger = setup_logger('my_script_name', level=logging.DEBUG)
# logger.info("This is an info message.")
# logger.debug("This is a debug message.") 