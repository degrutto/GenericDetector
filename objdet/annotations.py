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
import pycocotools.mask as mask_util
import PIL.Image as Image
import torch
from detectron2.structures.instances import Instances
import pprint

"""
Prodigy -> Detectron2 Adapter
"""


def add_color(segment_id, semcategory_ids, offset, space=1):
    if segment_id not in semcategory_ids.values():
        return [255, 255, 255]
    return np.array([0, 0, 0]) + np.array(
        [
            (segment_id + offset) * space,
            (segment_id + offset) * space,
            (segment_id + offset) * space,
        ]
    )


def segm_to_instances(segmentation, height, width):
    fields = {
        "scores": torch.tensor([0] * len(segmentation)),
        "gt_masks": torch.tensor(
            [mask_util.encode(segm["bit_mask"]) for segm in segmentation]
        ).to("cuda:0"),
        "classes": toch.tensor([segm["id"] for segm in segmentation]).to("cuda:0"),
        "boxes": torch.tensor([segm["bbox"] for segm in segmentation]).to("cuda:0"),
    }

    return Instances(((height, width)), **fields)


def convert_detection_to_panoptic_coco_format(
    record, category_ids, semcategory_ids, segmentations_folder
):

    print("using segmentations_folder ", segmentations_folder)
    pan_format = np.full((record["height"], record["width"], 3), 255, dtype=np.uint8)

    overlaps_map = np.zeros((record["height"], record["width"]), dtype=np.uint32)
    panoptic_record = {}
    panoptic_record["image_id"] = record["image_id"]
    panoptic_record["height"] = record["height"]
    panoptic_record["width"] = record["width"]
    file_name = "{}.png".format(Path(record["file_name"]))

    # print('file_name', file_name)

    segments_info = []
    anns = record["annotations"]
    panoptic_record_categories = []
    anns_updated = []
    # print("category_ids, semcategory_ids ", category_ids, semcategory_ids)
    # print("annotations", anns)
    # idxs_to_rm = []
    for idx, ann in enumerate(anns):
        # print("dealing with ann, idx", idx, ann)
        segment_id = ann["category_id"]
        color_id = add_color(segment_id, semcategory_ids, offset=0)
        panoptic_record_category = {}
        panoptic_record_category["id"] = color_id[0]
        panoptic_record_category["name"] = "Others"
        panoptic_record_category["color"] = color_id
        panoptic_record_category["supercategory"] = "damage"
        # [
        #    k for k, v in semcategory_ids.items() if v == ann["category_id"]
        # ][0]
        panoptic_record_category["is_thing"] = 1
        panoptic_record_category["category_id"] = 255
        for strcat, idcat in semcategory_ids.items():  ###  stuff
            # print("annotations: ", strcat, idcat)
            if idcat == segment_id:  # and strcat in semcategory_ids.keys():
                # print(strcat, idcat, " in semcategory_ids ")
                mask = polys_to_mask(
                    ann["segmentation"], record["height"], record["width"]
                )
                bit_mask = np.asfortranarray(mask, dtype=np.uint32)
                overlaps_map += bit_mask
                panoptic_record_category["is_thing"] = 0
                panoptic_record_category["category_id"] = color_id[0]
                panoptic_record_category["name"] = strcat
                # ann["bit_mask"] = bit_mask
                pan_format[bit_mask == 1] = color_id
                panoptic_record_category["supercategory"] = strcat
                # ann.pop("segmentation")
                # ann.pop("image_id")
                ann["id"] = color_id[0]
                segments_info.append(ann)
                # idxs_to_rm.append(idx)
        #        anns_updated.append(ann)

        panoptic_record_categories.append(panoptic_record_category)

    if np.sum(overlaps_map > 1) != 0:
        print("Segments for image {} overlap each other.".format(record["file_name"]))
        # return None

    # print("before delete, ", len(anns))
    # anns = [e for i, e in enumerate(anns) if i not in idxs_to_rm]
    # print(len(anns))
    panoptic_record["segment_info"] = segments_info
    # panoptic_record['instances'] = segm_to_instances(segments_info, height= record['height'], width = record['width'])
    panoptic_record["categories"] = panoptic_record_categories

    panoptic_record["annotations"] = anns  # _updated

    panoptic_record["file_name"] = record["file_name"]
    panoptic_record["pan_seg_file_name"] = os.path.join(
        segmentations_folder, os.path.basename(file_name)
    )
    panoptic_record["sem_seg_file_name"] = os.path.join(
        segmentations_folder, os.path.basename(file_name)
    )

    Image.fromarray(pan_format).save(
        os.path.join(segmentations_folder, os.path.basename(file_name))
    )
    # print(Image.shape)
    # print("np.array(Image.open(os.path.join(segmentations_folder, file_name)))", np.array(Image.open(os.path.join(segmentations_folder, file_name))).shape )
    # panoptic_record['sem_seg'] =  torch.from_numpy(np.array(Image.open(os.path.join(segmentations_folder, file_name)))).long().squeeze()[:,:,0]
    # print("converting coco annotations into record")  # , panoptic_record)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(panoptic_record)
    return panoptic_record


def polys_to_mask(polygons, height, width):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
    rle = mask_util.frPyObjects(polygons, height, width)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


@dataclass
class Annotation:
    """z
    Interface with Prodigy JSON file
    """

    path: Optional[Path]
    width: Optional[int]
    height: Optional[int]
    spans: Optional[List[Dict]]
    image_id: Optional[str]
    model_type: Optional[str]

    def as_datasetcatalog_dict(self, category_ids: Dict, semcategory_ids: Dict):
        record = dict()
        record["file_name"] = self.path
        record["image_id"] = self.image_id
        record["height"] = self.height
        record["width"] = self.width

        annos = list()
        for span in self.spans:
            _points = span["points"]
            _px = [x for x, y in _points]
            _py = [y for x, y in _points]
            bbox = [np.min(_px), np.min(_py), np.max(_px), np.max(_py)]
            bbox_mode = BoxMode.XYXY_ABS
            try:
                category_id = category_ids[span["label"]]
            except KeyError:
                print(f"Key {span['label']} not found in category_ids ... skipping")
                continue

            segmentation = [list(itertools.chain(*_points))]

            ### if we use bitmask as INPUT.MASK_FORMAT
            # mask = polys_to_mask(segmentation, self.height, self.width)
            # assert mask.sum()>0
            # bit_mask = np.asfortranarray(mask, dtype=np.uint8)
            # segmentation=mask_util.encode(bit_mask)

            assert len(_points) > 2
            anno_dict = dict(
                bbox=bbox,
                bbox_mode=bbox_mode,
                category_id=category_id,  # hack to insert label after
                segmentation=segmentation,
            )
            annos.append(anno_dict)

        record["annotations"] = annos

        if self.model_type == "panoptic_segmentation":
            segmentations_folder = os.path.join(
                Path(record["file_name"]).parent, "segmentations_folder"
            )
            if not os.path.isdir(segmentations_folder):
                print(
                    "Creating folder {} for panoptic segmentation PNGs".format(
                        segmentations_folder
                    )
                )
                os.mkdir(segmentations_folder)
            # print("original record is ", record)
            record = convert_detection_to_panoptic_coco_format(
                record, category_ids, semcategory_ids, segmentations_folder
            )

        # print("using record, " , record)
        return record

    def visualize_segmentation(self, category_ids):
        record = self.as_datasetcatalog_dict(category_ids)
        # print("visualize_segmentation:: record", record)
        # ims = []
        image = cv2.imread(
            record["file_name"][0]
        )  # np.zeros(shape=(record["height"],record["width"],3), dtype=np.uint8)
        # print(image.shape)
        for enum, segm in enumerate(record["annotations"]):
            try:
                # print("segm['segmentation']", segm["segmentation"])
                cid = segm["category_id"]
                # print("category id", cid)
                lab = [k for k, v in category_ids.items() if v == cid][0]

                segm = segm["segmentation"][0]

                # print("segm", segm)
                segm = np.array(
                    [[segm[i], segm[i + 1]] for i in np.arange(0, len(segm), 2)],
                    np.int32,
                )
                # print("seg fault here ? segm", segm)
                segm = segm.reshape((-1, 1, 2))
                # print("segm shape", segm.shape)
                # print(segm)
                # print("seg fault here ??")
                # print("adding original image")
                image = cv2.polylines(image, [segm], True, (255, 0, 0), 2)
                # else:
                # print("adding black image")
                # image = np.zeros(
                #    shape=(record["height"], record["width"], 3), dtype=np.uint8
                # )
                # im = cv2.polylines(image, [segm], True, (255, 0, 0), 2)
                # print("seg fault here ???")
                # print(im)
                # print("segm[0]", segm[0])
                print(category_ids)

                #                print(category_ids[record["category_id"]])
                image = cv2.putText(
                    image,
                    lab,
                    (segm[0][0][0], segm[0][0][1]),
                    cv2.FONT_ITALIC,
                    2,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            except Exception:
                print("we have an expection here")
                continue
            # print(image.sum())
        #           ims.append(im)
        #        print(ims)

        return image


@dataclass
class Annotations:

    anno_path: Path
    path_fix: Optional[Path]
    model_type: str

    def get_annotations(self) -> List[Annotation]:
        annotations = list()
        count = 0
        with open(os.path.join(self.anno_path), "r") as jsonl_file:
            json_list = list(jsonl_file)
            for json_str in tqdm(json_list):
                _anno = json.loads(json_str)

                if _anno["answer"] == "accept" and (
                    os.path.exists(_anno["path"])
                    or os.path.exists(self.pathfix(_anno["path"]))
                ):
                    if "spans" in _anno:
                        anno = Annotation(
                            path=self.pathfix(_anno["path"]),
                            width=_anno["width"],
                            height=_anno["height"],
                            spans=_anno["spans"],
                            image_id=count,
                            model_type=self.model_type,
                        )
                        count += 1
                        annotations.append(anno)

        return annotations

    def as_datasetcatalog_dicts(
        self,
        annotations: List[Annotation],
        category_ids: Dict,
        semcategory_ids: Dict,
        skips: Optional[List[int]] = [],
    ) -> List[Dict]:

        records = list()
        with tqdm(
            desc="Process annotations", total=len(annotations), mininterval=10
        ) as pbar:
            for i, anno in enumerate(annotations):
                # print(f"anno # {i}")
                if i in skips:
                    continue
                pbar.update(1)
                record = anno.as_datasetcatalog_dict(category_ids, semcategory_ids)
                if record is not None:
                    records.append(record)

        return records

    def pathfix(self, path: str) -> Path:
        path = Path(path)
        image_name = os.path.basename(path)
        fix = os.path.join(self.path_fix, image_name)
        return fix
