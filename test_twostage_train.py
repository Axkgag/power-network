import os

from task.classification.effnet import effnet_trainer
from task.classification.mulnet import mulnet_trainer
from task.object_detection.yolox import yolox_trainer
from task.object_detection.centernet import centernet_trainer
from task.two_stage.obj_cls import twostage_trainer
from utils.logsender import LogSender
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_dir',
                    default='../../ChipCounter/data/chip/twostage/test/model/')
parser.add_argument('-t', '--train_dir',
                    default='../../ChipCounter/data/chip/obj_cls/train/')
# parser.add_argument('-p', '--predictor_dir',
#                     default='../../ChipCounter/data/chip/datasets/wildanimals/train/yolox/')
# parser.add_argument('-c', '--classifier_dir',
#                     default='../../ChipCounter/data/chip/datasets/wildanimals/train/effnet/')

if __name__ == "__main__":
    args = parser.parse_args()
    train_dir = args.train_dir
    model_dir = args.model_dir

    mysender = LogSender()
    predictor = yolox_trainer.createInstance(mysender)
    classifier = effnet_trainer.createInstance(mysender)
    # predictor = centernet_trainer.createInstance(mysender)
    # classifier = mulnet_trainer.createInstance(mysender)
    trainer = twostage_trainer.createInstance(mysender, predictor, classifier)

    trainer.beginTrain(train_dir)
    # os.system("pause")

    # trainer.pauseTrain()
    # os.system("pause")
    #
    # trainer.resumeTrain()
    # os.system("pause")
    #
    # trainer.pauseTrain()
    # os.system("pause")
    #
    # trainer.resumeTrain()
    # os.system("pause")
    #
    # trainer.endTrain()
    # os.system("pause")
    # trainer.exportModel(model_dir)
