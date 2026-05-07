import yaml
import os
from .logger import setup_logger  


bootstrap_logger = setup_logger(config=None)

def load_config(config_path="configs/config.yaml"):
    
    
    
    if not os.path.exists(config_path):
        error_msg = f"Configuration file not found at: {config_path}"
        bootstrap_logger.error(error_msg)  
        raise FileNotFoundError(error_msg)
        
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            
        bootstrap_logger.info(f"Successfully loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as exc:
     
        error_msg = f"Failed to parse YAML file {config_path}. Error: {exc}"
        bootstrap_logger.error(error_msg)  
        raise
    except Exception as exc:
        
        error_msg = f"Unexpected error while reading {config_path}: {exc}"
        bootstrap_logger.critical(error_msg)  
        raise