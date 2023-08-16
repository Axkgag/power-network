import argparse

pretrain_model = {
    # (depth, width)
    "yolox_s": (0.33, 0.5),
    "yolox_m": (0.67, 0.75),
    "yolox_l": (1.0, 1.0),
    "yolox_x": (1.33, 1.25)
}

class TrainArgs:
    def __init__(self):
        self.name = "yolox_s"
        self.batch_size = 1
        self.devices = 0
        self.ckpt = "./weights/yolox_s.pth"
        self.resume = False
        self.start_epoch = 0
        self.experiment_name = "yolox_s"
        self.dist_backend = "nccl"
        self.dist_url = None
        self.num_machines = 1
        self.machine_rank = 0
        self.fp16 = False
        self.cache = False
        self.occupy = True
        self.opts = None
        self.dynamic = False
        self.decode_in_inference = False
        self.no_onnxsim = False

class PredictArgs:
    def __init__(self):
        self.name = "yolox_s"
        self.conf = 0.5
        self.nms = 0.4
        self.tsize = 640
        self.experiment_name = "yolox_s"
        self.ckpt = None
        self.save_result = True
        self.device = "gpu"
        self.fp16 = False
        self.legacy = False
        self.fuse = False
        self.trt = False

class EvalArgs:
    def __init__(self):
        self.name = "yolox_s"
        self.batch_size = 8
        self.conf = 0.5
        self.nms = 0.4
        self.tsize = 640
        self.ckpt = None
        self.experiment_name = "yolox_s"
        self.dist_backend = "nccl"
        self.dist_url = None
        self.devices = 0
        self.num_machines = 1
        self.machine_rank = 0
        self.seed = None
        self.fp16 = False
        self.fuse = False
        self.trt = False
        self.legacy = False
        self.test = False
        self.speed = False
        self.opts = None
