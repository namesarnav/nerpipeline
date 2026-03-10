

import logging




def setup_logger() -> logging.Logger:
    
    logger = logging.getLogger("ner_inference")
    
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
    
        fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    
        ch  = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)


    return logger
