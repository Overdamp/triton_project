import logging
import logging.handlers
import os

def setup_logger(config=None):

    

    if config is None:
        config = {
            'system': {
                'log_level': 'INFO',
                'log_file': 'logs/bootstrap.log' 
            }
        }
    
    system_config = config.get('system', {})
    log_level_str = system_config.get('log_level', 'INFO').upper()
    base_log_file = system_config.get('log_file', 'logs/system.log')
    
    log_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = log_levels.get(log_level_str, logging.INFO)
   
    os.makedirs(os.path.dirname(base_log_file), exist_ok=True)

    logger = logging.getLogger("AIPipeline")
    logger.setLevel(level) 
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # =========================================================
    # Handler 1: Console output (Standard Output)
    # =========================================================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # =========================================================
    # Handler 2: Main File Log (All logs up to configured level)
    # =========================================================
    main_file_handler = logging.handlers.RotatingFileHandler(
        filename=base_log_file, 
        maxBytes=5 * 1024 * 1024,  
        backupCount=10,            
        encoding='utf-8'
    )
    main_file_handler.setLevel(level)
    main_file_handler.setFormatter(formatter)
    logger.addHandler(main_file_handler)

    # =========================================================
    # Handler 3: Error File Log (Only ERROR and CRITICAL)
    # =========================================================
    file_name, file_extension = os.path.splitext(base_log_file)
    error_log_file = f"{file_name}_error{file_extension}"
    
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file, 
        maxBytes=5 * 1024 * 1024, 
        backupCount=5, 
        encoding='utf-8'
    )
    error_file_handler.setLevel(logging.ERROR) 
    error_file_handler.setFormatter(formatter)
    logger.addHandler(error_file_handler)

    return logger