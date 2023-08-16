import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def make_parser():
    parser = argparse.ArgumentParser("COCO evaluation parser")

    parser.add_argument("-r",
                        "--results",
                        dest="res",
                        default="../../datasets/train_collect/test/results.json",
                        type=str,
                        help="results.json")

    parser.add_argument("-a",
                        "--annos",
                        dest="annos",
                        default="../../datasets/train_collect/test/small_annotations.json",
                        type=str,
                        help="annotations.json")

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    coco = COCO(args.annos)
    coco_dt = coco.loadRes(args.res)

    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.catIds = [3]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

