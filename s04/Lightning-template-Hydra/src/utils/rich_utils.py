from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from loguru import logger
import sys

def setup_logger():
    """Setup loguru logger with rich console sink"""
    logger.remove()
    console = Console()
    
    # Add rich console as a sink
    logger.add(
        console.print,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file logging
    logger.add("logs/file_{time}.log", rotation="500 MB")
    
    return logger

def create_rich_progress_bar():
    """Create a rich progress bar for inference"""
    return Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=Console()
    ) 