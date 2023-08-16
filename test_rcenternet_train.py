from time import sleep
from task.object_detection.rcenternet import rcenternet_trainer
from utils.logsender import LogSender
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='../../ChipCounter/data/chip/')
parser.add_argument('-t', '--train_dir',
                    default='../../ChipCounter/data/chip/test/crop/')

if __name__=="__main__":
    mysender=LogSender()
    trainer = rcenternet_trainer.createInstance(mysender)

    args = parser.parse_args()
    model_dir = args.model_dir
    train_dir = args.train_dir
    
    trainer.beginTrain(train_dir)
    # os.system("pause")
    # trainer.exportModel(model_dir)
    # trainer.exportOnnx(model_dir, model_dir)
