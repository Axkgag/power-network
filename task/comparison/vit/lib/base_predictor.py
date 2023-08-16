import os

from .utils.config import Config

class Predictor():

    def __init__(self, config_path):
        self.config = Config(config_path)
        self.config_path=config_path
    
    def loadModel(self,model_dir_path):
        raise NotImplementedError

    
    def predict(self,img):
        return NotImplementedError