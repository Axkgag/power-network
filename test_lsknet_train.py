import os

from task.object_detection.lsknet import lsknet_trainer
from utils.logsender import LogSender
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='../../ChipCounter/data/chip/')
parser.add_argument('-t', '--train_dir',
                    default='/data/gauss/lyh/datasets/wildanimals/train')

if __name__ == "__main__":
    mysender = LogSender()
    trainer = lsknet_trainer.createInstance(mysender)

    args = parser.parse_args()
    model_dir = args.model_dir
    train_dir = args.train_dir

    trainer.beginTrain(train_dir)
    # os.system("pause")
    #
    # # trainer.pauseTrain()
    # # os.system("pause")
    # #
    # # trainer.resumeTrain()
    # # os.system("pause")
    #
    # trainer.endTrain()
    # os.system("pause")
    #
    # trainer.exportModel(model_dir)
    # trainer.exportOnnx(model_dir, model_dir)
    #
    # trainer.verify_onnx_model(model_dir, model_dir)
