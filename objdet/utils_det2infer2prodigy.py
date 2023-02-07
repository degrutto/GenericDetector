import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import Generator
import math
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures.boxes import Boxes
from axa.utils.masks_base import MaskBase
from detectron2.structures.instances import Instances

from axa.mrcnn.model import Model
import axa.mrcnn.utils_data as utils_data
import axa.mrcnn.utils_train as utils_train
from axa.mrcnn.kvisualizer import KVisualizer

import torch
from typing import List


from typing import Optional, Dict, Union, List, Callable, Generator

from axa.mrcnn.infer import select
from axa.mrcnn.masks_visualize import (
    predict,
    init_io,
    setup_det2_model,
    draw_predictions,
)


def build_predictor(
    model_dir: Path,
    score_threshold: float,
    model_type: str = "DAMAGE_MASKS",
    nms_threshold: float = 0.5,
):

    model_file, model_dir = init_io(model_dir)

    model = Model(model_type)
    model.log()

    cfg, metadata = setup_det2_model(
        model, model_dir, model_file, nms_threshold, score_threshold
    )

    return DefaultPredictor(cfg), metadata


def build_prodigy_dict(input_img_path, predictor, metadata, visualize=False):

    print(input_img_path)

    # _, image = utils_data.get_record(
    #    idx, filename, model, masks_file="", keypoints_file="", camera_file="",
    # )

    image = cv2.imread(input_img_path)

    # Predictions
    preds = predict(predictor, image)
    payload = {}

    if len(preds.pred_classes.numpy()) > 0:
        print(
            "we found these classes predictions",
            [
                metadata.get("thing_classes", None)[pred_class]
                for pred_class in preds.pred_classes.numpy()
            ],
        )

        payload = preds_to_prodigy_dict(
            preds, input_img_path, colors=metadata.thing_colors, metadata=metadata
        )

        if visualize:
            # draw image
            draw_image = draw_predictions(image, metadata, preds, obfuscate)

            vis_filepath = os.path.join(
                os.getcwd(), "visualizepred_" + os.path.basename(input_img_path)
            )

            cv2.imwrite(vis_filepath, draw_image)

    print("returning the payload ", payload)
    return payload


def preds_to_prodigy_dict(instances, image_path, colors: Optional[Dict], metadata):
    spans = list()
    lab_classes = [
        metadata.get("thing_classes", None)[pred_class]
        for pred_class in instances.pred_classes.numpy()
    ]

    for pclass in lab_classes:

        instances = select(
            instances,
            metadata.thing_classes,
            pclass,
            has_keypoints=False,
            highest_score=True,
        )

        for binary_mask in np.array(instances.pred_masks):

            contour = bitmask_to_contour(binary_mask)

            # segmentation_list = self.contour_to_segmentation_list(contour)
            points = segmentation_list_to_pairs(contour)

            span = dict(
                label=pclass,
                color=colors[lab_classes.index(pclass)],
                points=points,
                type="polygon",
            )
            spans.append(span)

    image = cv2.imread(image_path)
    payload = dict(
        image=image_path, width=image.shape[0], height=image.shape[1], spans=spans
    )
    return payload


def bitmask_to_contour(
    binary_mask_input: np.ndarray, _epsilon: Optional[float] = 0.004
):
    binary_mask_input = (binary_mask_input * 255).astype(np.uint8)

    _, binary_mask_input = cv2.threshold(binary_mask_input, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary_mask_input, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contour = largest_contour(contours)

    epsilon = _epsilon * cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx


def largest_contour(contours):
    return sorted(contours, key=lambda x: x.shape[0], reverse=True)[0]


def segmentation_list_to_pairs(contour):
    segm_list = contour.ravel().tolist()
    return [[segm_list[i], segm_list[i + 1]] for i in np.arange(0, len(segm_list), 2)]
