import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import numpy
def getAP(predict_filename,review_filename,iou_type):
    '''
    iou_type='bbox' or 'segm'
    '''
    review_anno=coco.COCO(review_filename)
    predict_anno=coco.COCO(predict_filename)
    return actualGetAP(gt_anno=review_anno,dets_anno=predict_anno,iou_type=iou_type)

def actualGetAP(gt_anno,dets_anno,iou_type):#coco_anno, coco_dets
    cat_info=gt_anno.dataset['categories']
    print("computing for info",cat_info)
    new_eval =COCOeval(cocoGt=gt_anno,cocoDt=dets_anno,iouType=iou_type)
    result={}
    new_eval.evaluate()
    new_eval.accumulate()
    new_eval.summarize()
    stats=new_eval.stats.tolist()
    result["map"]=stats
    result["class_ap_list"]=[]
    
    for info in cat_info:
        print("computing for info",info)
        new_eval.params.catIds=[info["id"]]
        new_eval.evaluate()
        new_eval.accumulate()
        # new_eval.summarize()
        stats=new_eval.stats.tolist()
        result["class_ap_list"].append({"class_name":info["name"],"ap":stats})
    return result

def getAPForRes(res_filename,origin_filename,iou_type):
    print("get ap for res and ori",res_filename,origin_filename)
    gt_anno=coco.COCO(origin_filename)
    dets_anno=gt_anno.loadRes(res_filename)
    return actualGetAP(gt_anno,dets_anno,iou_type)
        
def getMask(review_filename):
    import pycocotools.mask as maskUtils
    
    print(maskUtils.decode({"size": [1712, 2006], "counts": "PiiX3"}))

if __name__ =="__main__" : 
    # result= getAP('../../test-folder/eval/predict.json','../../test-folder/eval/review.json','segm')
    # print("result",json.dumps( result))
    getMask('../../test-folder/eval/mask.json')