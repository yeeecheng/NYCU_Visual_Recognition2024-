from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_mAP(gt_path, pred_path):
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # AP@[0.50:0.95]
    return coco_eval.stats[0]


