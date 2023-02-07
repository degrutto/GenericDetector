### USAGE  python scripts/visualizeProdigyAnnotations.py  --annotations annotations_1625238548.8308449.jsonl --imgsbasepath 2021


import argparse
import json
import os
from shutil import copyfile
from tqdm import tqdm
from objdet.annotations import Annotations
import os
import cv2


def visualize(jsonl_file, imgs_base_path):
    annos = Annotations(anno_path=jsonl_file, path_fix=imgs_base_path)
    annotations = annos.get_annotations()[:1]
    category_ids = {
        "Scratch": 0,
        "Crack": 1,
        "Bump": 2,
        "Broken": 3,
        "Gap": 4,
        "Plate": 5,
        "Face": 6,
    }
    for ea, anno in enumerate(annotations):
        print(anno)
        im = anno.visualize_segmentation(category_ids)
        # print(im, im.shape, im.sum())
        cv2.imwrite(f"img_{ea}.jpg", im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass a directory with Prodigy jsons")

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

    args = parser.parse_args()
    visualize(jsonl_file=args.annotations, imgs_base_path=args.imgsbasepath)
