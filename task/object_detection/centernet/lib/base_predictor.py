import os
import sys
sys.path.append(".")
from .utils.config import Config
from abc import ABCMeta, abstractmethod

class Predictor():

    def __init__(self, config_path):
        self.config = Config(config_path)
    
    @abstractmethod
    def loadModel(self,model_path):
        raise NotImplementedError

    @abstractmethod
    def predict(self,img):
        return NotImplementedError