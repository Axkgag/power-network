import numpy as np
import time
import torch


from ..models.decode import ctdet_decode
from ..utils.post_process import ctdet_post_process

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]
  
  def stack_post_process(self, dets, metas, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = ctdet_post_process(
        dets.copy(), metas['c'], metas['s'],
        metas['out_height'], metas['out_width'], self.opt.num_classes) # [{class_id: numpy.array([[x, y, x, y, score]])}, {}, ]
    for i in range(len(dets)):
      for j in range(1, self.num_classes + 1):
        dets[i][j] = np.array(dets[i][j], dtype=np.float32).reshape(-1, 5)
        dets[i][j][:, :4] /= scale
    return dets # [{class_id: numpy.array([[x, y, x, y, score]])}, {}, ]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results
  
  def stack_merge_outputs(self, dets):
    # dets: # [{class_id: numpy.array([[x, y, x, y, score]]), class_id: -}, {}, ]
    results_list = []
    for det in dets:
      results = {}
      for j in range(1, self.num_classes + 1):
        results[j] = det[j]
      scores = np.hstack(
        [results[j][:, 4] for j in range(1, self.num_classes + 1)])
      if len(scores) > self.max_per_image:
        kth = len(scores) - self.max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, self.num_classes + 1):
          keep_inds = (results[j][:, 4] >= thresh)
          results[j] = results[j][keep_inds]
      results_list.append(results)
    return results_list
