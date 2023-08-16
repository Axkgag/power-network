from task.classification.mulnet import mulnet_trainer
from utils.sender import Sender
import argparse
import time
parser=argparse.ArgumentParser()
parser.add_argument('-m','--model_dir', default='shared_dir/train_result_dir/mulnet_model', # 需要
                             help='model_dir, dir of model to be loaded')
parser.add_argument('-t','--train_dir', default='/data/gauss/lyh/datasets/power_networks/mulnet', # 需要
                        help='train_dir, dir of dataset be train')
if __name__=="__main__":
    mysender=Sender("localhost",3005)
    args=parser.parse_args()
    model_dir=args.model_dir
    train_dir=args.train_dir
    trainer=mulnet_trainer.createInstance(mysender)
    trainer.beginTrain(train_dir)
    # time.sleep(10)
    # trainer.pauseTrain()
    # time.sleep(10)
    # trainer.resumeTrain()
    # time.sleep(10)
    # trainer.endTrain()
