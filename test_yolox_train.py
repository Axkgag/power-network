import os

from task.object_detection.yolox import yolox_trainer
from utils.logsender import LogSender
import argparse
from test_yolox_predict import test

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train_dir',
                    default="./datasets/grid/")
parser.add_argument('-c', '--train_config',
                    default='config/trainer.json')
parser.add_argument('-r', '--is_resume',
                    action="store_true",
                    default=False)
parser.add_argument('-m', '--test_mode',
                    type=str,
                    default=None)

if __name__ == "__main__":
    mysender = LogSender()

    args = parser.parse_args()
    train_dir = args.train_dir
    train_config = args.train_config
    is_resume = args.is_resume
    test_mode = args.test_mode

    trainer = yolox_trainer.YoloxTrainer(mysender, train_config)

    trainer.beginTrain(train_dir, is_resume=is_resume)

    if test_mode:
        test(train_dir, trainer.export_dir, False, test_mode)
    # nohup python test_yolox_train.py -t /data/gauss/lyh/datasets/power_networks/yolox -c config/trainer_1.json > /data/gauss/lyh/datasets/power_networks/model/ensamble/model1/output.log 2>&1 &
