import os

from task.object_detection.yolox import yolox_trainer
from utils.logsender import LogSender
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_dir',
                    default="")
parser.add_argument('-c', '--train_config',
                    default='config/trainer.json')

if __name__ == "__main__":
    mysender = LogSender()

    args = parser.parse_args()
    train_dir = args.train_dir
    train_config = args.train_config

    trainer = yolox_trainer.YoloxTrainer(mysender, train_config)

    trainer.beginTrain(train_dir, is_resume=False)

    # nohup python test_yolox_train.py -t /data/gauss/lyh/datasets/power_networks/yolox -c config/trainer_1.json > /data/gauss/lyh/datasets/power_networks/model/ensamble/model1/output.log 2>&1 &
