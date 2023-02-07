### usageF python scripts/trainer.py --annotations annotations_1625238548.8308449.jsonl  --imgsbasepath /beegfs/gva.inaitcloud.com/projects/object_detection_results/ImagesForAnnotators_June2021/  --model65;6003;1c65;6003;1c65;6003;1ctype "instance_segmentation" --classes Scratch Dent Broken Plate Face Gap Crack

import argparse
from pathlib import Path
from objdet.annotations import Annotations
from objdet.modelconfig import ModelConfig
from objdet.LossEvalHook import LossEvalHook

import os
import time
import cv2
import torch
import shutil
import subprocess
import numpy as np
import random

from typing import Optional, List, Dict

# from objdet.kvisualizer import KVisualizer
from objdet.plotmetrics import plot_losses
import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.engine.hooks import HookBase
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    SemSegEvaluator,
    DatasetEvaluator,
)
from pprint import pformat
from datetime import datetime
import detectron2.data.transforms as T


from detectron2.data.datasets import register_coco_panoptic
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        # return build_detection_train_loader(cfg, mapper=cls.mapper)
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=build_sem_seg_train_aug(cfg),
                use_instance_mask=cfg.MODEL.MASK_ON,
                instance_mask_format=cfg.INPUT.MASK_FORMAT,
                recompute_boxes=False,
            ),
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            # if cfg['modeltype'] == "panoptic_segmentation":
            #   return    SemSegEvaluator(dataset_name, output_dir=output_folder)
            # else :
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],
                    mapper=DatasetMapper(
                        self.cfg,
                        True,
                        use_instance_mask=self.cfg.MODEL.MASK_ON,
                        instance_mask_format=self.cfg.INPUT.MASK_FORMAT,
                        recompute_boxes=False,
                    ),
                ),
            ),
        )
        return hooks


def build_sem_seg_train_aug(cfg):

    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        augs.append(T.RandomBrightness(0.9, 1.1))
        augs.append(T.RandomRotation([-30, 30]))
        augs.append(T.RandomContrast(0.9, 1.1))

    if cfg.INPUT.RANDOM_FLIP != "none":
        augs.append(T.RandomFlip())

    print(f"build_train_loader:: build_sem_seg_train_aug, using augs {augs}")

    return augs


def make_dataset(
    annos: Annotations,
    modeltype: str,
    category_ids: Dict,
    semcategory_ids: Dict,
    train_split: Optional[float] = 0.9,
):
    """
    Register the dataset
    Args:
        path: path to JSON file containing the Prodigy Annotations
        category_ids: Dict containing the label string -> ID mapping
    """

    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.clear()
    MetadataCatalog.clear()
    annotations = annos.get_annotations()

    nb_training = int(train_split * len(annotations))
    print("nb_training", nb_training)
    train_split = annotations[:nb_training]
    test_split = annotations[nb_training:]
    ### work in better split here

    if modeltype == "panoptic_segmentation":
        MetadataCatalog.get("train_data").set(
            thing_classes=list(category_ids.keys()),
            stuff_classes=list(semcategory_ids.keys()),
            thing_dataset_id_to_contiguous_id=dict(
                [(i, i) for i in category_ids.values()]
            ),
            stuff_dataset_id_to_contiguous_id=dict(
                [(i, i) for i in semcategory_ids.values()]
            ),
            ignore_label = 255
        )
        MetadataCatalog.get("test_data").set(
            thing_classes=list(category_ids.keys()),
            stuff_classes=list(semcategory_ids.keys()),
            thing_dataset_id_to_contiguous_id=dict(
                [(i, i) for i in category_ids.values()]
            ),
            stuff_dataset_id_to_contiguous_id=dict(
                [(i, i) for i in semcategory_ids.values()]
            ),
            ignore_label = 255
        )
    else:

        MetadataCatalog.get("train_data").set(thing_classes=list(category_ids.keys()))

        MetadataCatalog.get("test_data").set(thing_classes=list(category_ids.keys()))

    # if modeltype == "instance_segmentation":
    DatasetCatalog.register(
        "train_data",
        lambda: annos.as_datasetcatalog_dicts(
            train_split, category_ids, semcategory_ids
        ),
    )

    DatasetCatalog.register(
        "test_data",
        lambda: annos.as_datasetcatalog_dicts(
            test_split, category_ids, semcategory_ids
        ),
    )
    # if modeltype == "panoptic_segmentation" :
    #    instance_segmentation

    #    register_coco_panoptic_separated("train_data", image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json).

    print(f"DatasetCatalog, {DatasetCatalog,}")

    return DatasetCatalog, MetadataCatalog


def train(
    imgs_base_path: Path,
    modeltype: str,
    jsonl_file: Path,
    classes: List[str],
    semclasses: List[str],
    modelinitpath: Path,
) -> list:
    """ main function to do training """

    annos = Annotations(
        anno_path=jsonl_file, path_fix=imgs_base_path, model_type=modeltype
    )
    # annotations = annos.get_annotations()[:1]
    category_ids = {}
    semcategory_ids = {}
    for enum, lab in enumerate(classes):
        category_ids[lab] = enum

    semcategory_ids = {}
    for enum, lab in enumerate(semclasses):
        semcategory_ids[lab] = enum

    print(f"category_ids {category_ids}")
    cuda_device = torch.cuda.current_device()
    cuda_nb_devices = torch.cuda.device_count()
    cuda_available = bool(torch.cuda.is_available())
    print(
        f"Using CUDA: available={cuda_available}, nb_devices={cuda_nb_devices}, current_device={cuda_device}"
    )
    assert cuda_available

    datasetcat, metadatacat = make_dataset(
        annos, modeltype, category_ids, semcategory_ids
    )

    cfg = ModelConfig().cfg(
        modeltype,
        list(category_ids.keys()),
        list(semcategory_ids.keys()),
        modelinitpath,
    )
    cfg["modeltype"] = modeltype
    print(f"using config {cfg}")

    start_time = time.time()

    launch(
        main, torch.cuda.device_count(), dist_url="auto", args=(cfg,),
    )

    end_time = time.time() - start_time

    print("FULL RUN: ", end_time)

    plot_losses(cfg.OUTPUT_DIR, os.path.join(cfg.OUTPUT_DIR, "losses.png"))

    # visualization: orig + annot + preds
    from detectron2.utils.visualizer import ColorMode

    predictor = DefaultPredictor(cfg)
    testdatadict = datasetcat.get("test_data")
    metadata = metadatacat.get("test_data")
    for d in random.sample(testdatadict, min(10, len(testdatadict))):
        im = cv2.imread(d["file_name"])
        if modeltype == "instance_segmentation":

            outputs = predictor(im)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        elif modeltype == "panoptic_segmentation":
            panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

        img = out.get_image()[:, :, ::-1]
        # annot = v.draw_dataset_dict(d)
        # annot = annot.get_image()[:, :, ::-1]
        # cv2_imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite(
            os.path.join(cfg.OUTPUT_DIR, "vis_" + os.path.basename(d["file_name"])),
            np.hstack((im, img)),
        )


def main(cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set up inputs")

    parser.add_argument(
        "--modeltype",
        metavar="type of model",
        required=True,
        help="the type of model and specification, e.g. 'instance_segmantation'",
    )

    parser.add_argument(
        "--classes",
        metavar="classes",
        nargs="*",
        type=str,
        required=True,
        default=[],
        help="list of string of classes to train, ex ['scratch','bump']",
    )

    parser.add_argument(
        "--semclasses",
        metavar="semclasses",
        nargs="*",
        type=str,
        required=False,
        default=[],
        help="list of string of sem classes to train, ex ['scratch']",
    )

    parser.add_argument(
        "--annotations",
        metavar="path",
        required=True,
        help="the path to the annotations file",
    )

    parser.add_argument(
        "--imgsbasepath",
        metavar="imgsbasepath",
        required=True,
        help="the path to the imgs file",
    )

    parser.add_argument(
        "--modelinitpath",
        metavar="modelinitpath",
        required=False,
        default=None,
        help="optional, the path to the .pth file to initialize the model, if not provided the default is model zoo",
    )

    args = parser.parse_args()
    train(
        jsonl_file=args.annotations,
        imgs_base_path=args.imgsbasepath,
        modeltype=args.modeltype,
        classes=args.classes,
        semclasses=args.semclasses,
        modelinitpath=args.modelinitpath,
    )
