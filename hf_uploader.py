import huggingface_hub
from dotenv import load_dotenv
load_dotenv()

import os

HF_KEY = os.getenv('HF_KEY')

def upload_outputs_to_hub(path): 
    
    if not(path): 
        raise FileNotFoundError
    
    else:
        huggingface_hub.upload_file(path)