import logging
import os
from datetime import datetime

class SystemLogger:
    def __init__(
        self, 
        log_dir='logs', 
        log_level=logging.INFO
    ):
        """
        Initialize a comprehensive logging system
        
        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level
        """
        # Create logs directory if not exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'classification_{timestamp}.log')
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ClassificationSystem')
    
    def info(self, message):
        """Log informational message"""
        self.logger.info(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_model_performance(self, metrics):
        """
        Log model performance metrics
        
        Args:
            metrics (dict): Performance metrics
        """
        self.logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

# Global logger instance
system_logger = SystemLogger() 