import os
import sys
sys.path.append(os.path.join(os.getcwd(),"lib"))

from .utils.config import Config


class Trainer(object):

    def __init__(self, config_path):
        self.config = Config(config_path)
        self.stop_signal = False
    
    def beginTrain(self):
        raise NotImplementedError
    
    def pauseTrain(self):
        self.stop_signal = True

    def resumeTrain(self):
        self.beginTrain()
    
    def endTrain(self):
        self.stop_signal = True
    
    def sendTrainStatus(self):
        raise NotImplementedError
    
    def exportModel(self):
        raise NotImplementedError

    def testCurrentOnDataset(self):
        raise NotImplementedError