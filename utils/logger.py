import logging
import os
from rich.logging import RichHandler

def setup_logger(name="agent_logger", log_file="agent.log", level=logging.INFO):
    """
    Function to setup a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_file (str): Path to the log file
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logger without
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers are already added to avoid duplicate logs
    if not logger.handlers:
        # Create console handler using RichHandler for better formatting
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_handler.setLevel(level)
        
        # Create file handler
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            
            # Create formatter and add it to the file handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add the handlers to the logger
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")
            
        logger.addHandler(console_handler)
        
    return logger
