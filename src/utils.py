import httpx
from src.chat import MODEL
import logging

def server_ping() -> bool: 
    try:
        response = httpx.get("http://localhost:11434")
        return response.status_code == 200
    except:
        return False
    
def model_check() -> bool:
    try:
        response = httpx.get("http://localhost:11434/api/tags")
        for model in response.json()["models"]:
            if MODEL in model["name"]:
                return True
        return False
    except Exception as e:
        print("some error happened")
        logging.exception("message")
        return False
