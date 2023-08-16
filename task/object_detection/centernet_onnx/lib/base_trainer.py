
class Trainer(object):

    def __init__(self):
        pass
    
    def beginTrain(self):
        raise NotImplementedError
    
    def pauseTrain(self):
        raise NotImplementedError

    def resumeTrain(self):
        raise NotImplementedError
    
    def endTrain(self):
        raise NotImplementedError
    
    def sendTrainStatus(self):
        raise NotImplementedError
    
    def exportModel(self):
        raise NotImplementedError

    def testCurrentOnDataset(self):
        raise NotImplementedError