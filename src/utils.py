import os
import sys
import dill
import pickle

from src.exception import CustomException

def save_object(file_path: str, obj):
    """
    Save Python object to file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    """
    Load Python object from file using pickle.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)
