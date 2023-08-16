from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np

from torch._C import CONV_BN_FUSION

class opts(object):
  def __init__(self):
      # basic experiment setting
      self.task = 'ctdet'  # help='ctdet'
      self.dataset = 'coco'  # help='pascal'
      self.exp_id = 'default'
      self.test = False  # action='store_true'
      self.debug = 0  # help='level of visualization.'
      # '1: only show the final detection results'
      # '2: show the network output features'
      # '3: use matplot to display' # useful when lunching training with ipython notebook
      # '4: save all visualizations to disk')
      self.demo = ''  # help='path to image/ image folders/ video or "webcam"'
      self.load_model = ''  # help='path to pretrained model'
      self.resume = False  # action='store_true'

      # system
      self.gpus = '0'  # help='-1 for CPU, use comma for multiple gpus')
      self.num_workers = 4  # help='dataloader threads. 0 for single-thread.'
      self.not_cuda_benchmark = False  # help='disable when the input size is not fixed.'
      self.seed = 317  # help='random seed'

      # log
      self.print_iter = 0  # 'disable progress bar and print to screen.'
      self.hide_data_time = False  # help='not display time during training.'
      self.save_all = False  # help='save model to disk every 5 epochs.'
      self.metric = 'loss'  # help='main metric to save best model'
      self.vis_thresh = 0.3  # help='visualization threshold.'
      self.debugger_theme = 'white'  # choices=['white', 'black']

      # model
      self.arch = 'dla_34'  # help='model architecture. Currently tested'
      # 'res_18 | res_101 | resdcn_18 | resdcn_101 |'
      # 'dlav0_34 | dla_34 | hourglass')
      self.head_conv = -1  # help='conv layer channels for output head'
      # '0 for no conv layer'
      # '-1 for default setting: '
      # '64 for resnets and 256 for dla.')
      self.down_ratio = 4  # help='output stride. Currently only supports 4.'

      # input
      self.input_res = -1  # help='input height and width. -1 for default from '
      # 'dataset. Will be overriden by input_h | input_w'
      self.input_h = -1  # help='input height. -1 for default from dataset.'
      self.input_w = -1  # help='input width. -1 for default from dataset.'

      # train
      self.lr = 1.25e-4  # help='learning rate for batch size 32.'
      self.lr_step = '90,120'  # help='drop learning rate by 10.')
      self.num_epochs = 140  # help='total training epochs.'
      self.batch_size = 32  # help='batch size'
      self.master_batch_size = -1  # help='batch size on the master gpu.'
      self.num_iters = -1  # help='default: #samples / batch_size.'
      self.val_intervals = 5  # help='number of epochs to run validation.'
      self.trainval = False  # help='include validation in training and test on test set

      # test
      self.flip_test = False  # help='flip data augmentation.
      self.test_scales = '1'  # help='multi scale test augmentation.'
      self.nms = False  # help='run nms in testing.'
      self.K = 100  # help='max number of output objects.'
      self.not_prefetch_test = False  # help='not use parallal data pre-processing.'
      self.fix_res = False  # help='fix testing resolution or keep the original resolution
      self.keep_res = False  # help='keep the original resolution during validation.'

      # dataset
      self.not_rand_crop = False  # help='not use the random crop data augmentation from CornerNet.'
      self.shift = 0.1  # help='when not using random crop apply shift augmentation.'
      self.scale = 0.4  # help='when not using random crop apply scale augmentation.'
      self.rotate = 0  # help='when not using random crop apply rotation augmentation.'
      self.flip = 0.5  # help='probability of applying flip augmentation.'
      self.no_color_aug = False  # help='not use the color augmenation from CornerNet'

      # ctdet
      self.reg_loss = 'l1'  # help='regression loss: sl1 | l1 | l2'
      self.hm_weight = 1  # help='loss weight for keypoint heatmaps.'
      self.off_weight = 1  # help='loss weight for keypoint local offsets.'
      self.wh_weight = 0.1  # help='loss weight for bounding box size.'
      self.ang_weight = 0.1  # help='loss weight for ratation angle.'

      # loss
      self.mse_loss = False  # help='use mse loss or focal loss to train keypoint heatmaps.'

      # task
      # ctdet
      self.norm_wh = False  # help='L1(\hat(y) / y, 1) or L1(\hat(y), y)'
      self.dense_wh = False  # help='apply weighted regression near center or just apply regression on center point.'
      self.cat_spec_wh = False  # help='category specific bounding box size.'
      self.not_reg_offset = False  # help='not regress local offset.'

      # data dir
      self.img_dir = ''  # help='the directory of training data and test data.')
      self.train_annot_path = ''
      self.test_annot_path = ''
      self.output_dir = '../exp'
      self.num_classes = 80
      self.class_names = [
          'person', 'bicycle', 'car', 'motorcycle', 'airplane',
          'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
          'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
          'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
          'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
          'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
          'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
          'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
          'scissors', 'teddy bear', 'hair drier', 'toothbrush']

  # def parse(self, args=''):
  #   if args == '':
  #     opt = self.parser.parse_args()
  #   else:
  #     opt = self.parser.parse_args(args)
  #   return opt
  
  def decode(self, opt):
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    # opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
    opt.gpus = [i for i in opt.gpus] if opt.gpus[0] >= 0 else [-1]
    opt.lr_step = [float(i) for i in opt.lr_step.split(',')]
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

    opt.fix_res = not opt.keep_res
    print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
    opt.reg_offset = not opt.not_reg_offset

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64
    opt.pad = 127 if 'hourglass' in opt.arch else 31
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1

    if opt.trainval:
      opt.val_intervals = 100000000

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.exp_dir = os.path.join(opt.output_dir, opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, config):

    input_h, input_w = config["input_res"]
    if config['input_res']is not None:
      input_h, input_w = config['input_res']
    del config['input_res']
    opt_vars = vars(opt)
    for key in config:
      if config[key] is not None:
        opt_vars[key] = config[key]
        setattr(opt,key,config[key])
    print(opt)
    # if config.get('class_names') is not None:
    #   opt.class_names = config.get('class_names')
    # if config.get('num_classes') is not None:
    #   opt.num_classes = config.get('num_classes')
    # if config.get('img_dir') is not None:
    #   opt.img_dir = config.get('img_dir')
    # if config.get('train_annot_path') is not None:
    #   opt.train_annot_path = config.get('train_annot_path')
    # if config.get('val_annot_path') is not None:
    #   opt.val_annot_path = config.get('val_annot_path')
    # if config.get('test_annot_path') is not None:
    #   opt.test_annot_path = config.get('test_annot_path')
    # if config.get('output_dir') is not None:
    #   opt.output_dir = config.get('output_dir')
    # if config.get('task') is not None:
    #   opt.task = config.get('task')
    # if config.get('exp_id') is not None:
    #   opt.exp_id = config.get('exp_id')
    # if config.get('load_model') is not None:
    #   opt.load_model = config.get('load_model')
    # if config.get('resume') is not None:
    #   opt.resume = config.get('resume')
    # if config.get('gpus') is not None:
    #   opt.gpus = config.get('gpus')
    # if config.get('num_workers') is not None:
    #   opt.num_workers = config.get('num_workers')
    # if config.get('arch') is not None:
    #   opt.arch = config.get('arch')
    # if config.get('lr') is not None:
    #   opt.lr = config.get('lr')
    # if config.get('lr_step') is not None:
    #   opt.lr_step = config.get('lr_step')
    # if config.get('num_epochs') is not None:
    #   opt.num_epochs = config.get('num_epochs')
    # if config.get('batch_size') is not None:
    #   opt.batch_size = config.get('batch_size')
    # if config.get("val_intervals") is not None:
    #   opt.val_intervals = config.get("val_intervals")
    # if config.get("test_file") is not None:
    #   opt.test_file = config.get("test_file")
    # if config.get("test_model") is not None:
    #   opt.test_model = config.get("test_model")
    
    opt = self.decode(opt)

    # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
    input_h = opt.input_res if opt.input_res > 0 else input_h
    input_w = opt.input_res if opt.input_res > 0 else input_w
    opt.input_h = opt.input_h if opt.input_h > 0 else input_h
    opt.input_w = opt.input_w if opt.input_w > 0 else input_w
    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w)
    opt.output_res = max(opt.output_h, opt.output_w)
    
    if opt.task == 'ctdet':
      opt.heads = {'hm': opt.num_classes,
                   'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes, 'ang': 1}
      if opt.reg_offset:
        opt.heads.update({'reg': 2})
    else:
      assert 0, 'task not defined!'
    print('heads', opt.heads)
    return opt

  def init(self, config, args=''):
    default_dataset_info = {
      'ctdet': {'default_resolution': [512, 512],
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco'},
    }
    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    opt = self.parse(args)
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset, config)
    return opt
