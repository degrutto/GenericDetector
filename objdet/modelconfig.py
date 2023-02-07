
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
import itertools
import numpy as np
from detectron2.structures import BoxMode
from tqdm import tqdm
import cv2
from copy import deepcopy
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2.utils.comm as comm
import torch
import os
from datetime import datetime


@dataclass
class ModelConfig:
    """
    Interface with Prodigy JSON file
    """

    mtype: str = "instance_segmentation"
    model_things: Optional[List[Dict]] = "['scratch']"
    model_stuffs: Optional[List[Dict]] = "['scratch']"

    """ DataClass to build model config"""

    def cfg(self, mtype, model_things, model_stuffs, model_init_path):

        if mtype == "instance_segmentation":
            return self.instsegmcfg(model_things, model_init_path)
        elif mtype == "panoptic_segmentation":
            return self.panopticcfg(model_things, model_stuffs, model_init_path)
        else:
            return None

    def instsegmcfg(self, model_things, model_init_path):
        print(f"using model_init_path {model_init_path}")

        mask_model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(mask_model_config))

        if model_init_path is not None:
            cfg.MODEL.WEIGHTS = model_init_path  ## xxx/model_optim.pth
            print("inizializing weight from model {model_path}")
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mask_model_config)

        cfg.INPUT.MASK_FORMAT = "polygon"

        cfg.INPUT.CROP.ENABLED = True
        cfg.INPUT.RANDOM_FLIP = "horizontal"
        cfg.TEST.AUG.FLIP = False

        cfg.SOLVER.MAX_ITER = 10000
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.WARMUP_ITERS = 0

        cfg.TEST.EVAL_PERIOD = 1000
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD
        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

        cfg.MODEL.DEVICE = "cuda" if bool(torch.cuda.is_available()) else "cpu"

        cfg.SOLVER.BASE_LR = 0.0005
        cfg.SOLVER.GAMMA = 0.99

        cfg.TEST.EXPECTED_RESULTS = []
        cfg.DATASETS.TRAIN = ("train_data",)
        cfg.DATASETS.TEST = ("test_data",)
        cfg.INPUT.MAX_SIZE_TEST = 1000
        cfg.INPUT.MAX_SIZE_TRAIN = 1000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(model_things)
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 21

        cfg.OUTPUT_DIR = (
            "model_dir_instance" + f'{datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")}'
        )

        if comm.is_main_process:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
                f.write(cfg.dump())

        return cfg

    def panopticcfg(self, model_things, model_stuffs, model_init_path):
        print(f"using model_init_path {model_init_path}")

        mask_model_config = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(mask_model_config))

        if model_init_path is not None:
            cfg.MODEL.WEIGHTS = model_init_path  ## xxx/model_optim.pth
            print("inizializing weight from model {model_path}")
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(mask_model_config)

        cfg.INPUT.MASK_FORMAT = "polygon"

        cfg.INPUT.CROP.ENABLED = True
        cfg.INPUT.RANDOM_FLIP = "horizontal"
        cfg.TEST.AUG.FLIP = False

        cfg.SOLVER.MAX_ITER = 20000
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        cfg.SOLVER.STEPS = []
        cfg.SOLVER.WARMUP_ITERS = 0

        

        cfg.TEST.EVAL_PERIOD = 2000
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD

        cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

        cfg.MODEL.DEVICE = "cuda" if bool(torch.cuda.is_available()) else "cpu"

        cfg.SOLVER.BASE_LR = 0.0005
        cfg.SOLVER.GAMMA = 0.99

        cfg.TEST.EXPECTED_RESULTS = []
        cfg.DATASETS.TRAIN = ("train_data",)
        cfg.DATASETS.TEST = ("test_data",)
        cfg.INPUT.MAX_SIZE_TEST = 1000
        cfg.INPUT.MAX_SIZE_TRAIN = 1000
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(model_things) 
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(model_stuffs) +  1  # 2
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 21

        cfg.OUTPUT_DIR = (
            "model_dir_panoptic" + f'{datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")}'
        )

        if comm.is_main_process:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
                f.write(cfg.dump())

        return cfg
