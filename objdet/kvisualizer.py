import numpy as np
from enum import Enum
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from colorsys import hsv_to_rgb


def pseudocolor(val, minval, maxval):
    """Convert val in range minval..maxval to the range 0..120 degrees which
    correspond to the colors Red and Green in the HSV colorspace.
    """
    h = min(float(val - minval) / (maxval - minval), 1) * 120

    # Convert hsv color (h,1,1) to its rgb equivalent.
    # Note: hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = hsv_to_rgb(h / 360, 1.0, 1.0)
    return r, g, b


class KeyDrawEnum(Enum):
    ALL = 0  # Draw all keypoints
    LEFT_ONLY = 1  # Draw only the left keypoints
    RIGHT_ONLY = 2  # Draw only the right keypoints
    INLIERS_ONLY = 3  # Draw only the inlier keypoints


class KVisualizer(Visualizer):
    """ Visualizer with custom drawing of keypoints, depending on their visibility flag """

    inliers = []
    keydraw = KeyDrawEnum.ALL
    min_val = 0
    max_val = 10

    def draw_keypoint(self, x, y, v, idx, radius=4):
        """ """
        self.draw_circle(
            (x, y), color=pseudocolor(v, self.min_val, self.max_val), radius=radius
        )
        self.draw_text(
            f"{idx}",
            (x, y + radius),
            color=(1, 1, 0),
            font_size=self._default_font_size * 2 / 4,
        )

    def draw_and_connect_keypoints(self, keypoints):
        """ """
        for idx, keypoint in enumerate(keypoints):
            if self.keydraw == KeyDrawEnum.ALL:
                draw = True
            elif self.keydraw == KeyDrawEnum.LEFT_ONLY:
                draw = True if idx < 40 else False
            elif self.keydraw == KeyDrawEnum.RIGHT_ONLY:
                draw = True if idx >= 40 else False
            else:
                draw = idx in self.inliers

            if draw:
                x, y, v = keypoint
                self.draw_keypoint(x, y, v, idx)

        return self.output

    def create_text_labels(self, classes, scores, class_names):
        """
        Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

        Returns:
        list[str] or None
        """
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = [
                    "{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)
                ]

        return labels

    def create_text_labels_iou(
        self,
        classes,
        scores,
        ious,
        class_names,
        noious=False,
        depths=None,
        volumes=None,
    ):
        """
        Args:
        classes (list[int] or None):
        scores (list[float] or None):
        ious (list[float] or None):
        class_names (list[str] or None):

        Returns:
        list[str] or None
        """
        labels = None
        iouslab = None
        depthslab = None
        volumeslab = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["s {:.0f}%".format(s * 100) for s in scores]
                if not noious:
                    iouslab = ["iou {:.0f}%".format(s * 100) for s in ious]
                depthslab = ["" for d in scores]
                volumeslab = ["" for d in scores]
            else:
                labels = [
                    "{} s {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)
                ]
                if not noious:
                    iouslab = ["iou {:.0f}%".format(s * 100) for s in ious]

                if depths is not None:
                    depthslab = [
                        "width, high, depth[cm] {:.1f} {:.1f} {:.1f}".format(
                            d[0] * 100, d[1] * 100, d[2] * 100
                        )
                        if class_names[classes[i]] == "bump"
                        else "width, high {:.1f} {:.1f}".format(d[0] * 100, d[1] * 100)
                        for (i, d) in enumerate(depths)
                    ]

                if volumes is not None:
                    volumeslab = [
                        "volume[cm3] {:.1f}".format(d * 1000000)  # volumes are in m3
                        if class_names[classes[i]] == "bump"
                        else ""
                        for (i, d) in enumerate(volumes)
                    ]

        if iouslab:
            if volumeslab and depthslab:
                return list(zip(labels, iouslab, depthslab, volumeslab))
            else:
                return list(zip(labels, iouslab))

        if depthslab and depthslab:
            return list(zip(labels, depthslab, volumeslab))

        if depthslab:
            return list(zip(labels, depthslab))

        return labels

    def draw_instance_predictions(
        self,
        predictions,
        ious_m,
        nms_threshold=0.5,
        obfuscate=False,
        add_labels=None,
        noious=False,
        depths=None,
        volumes=None,
    ):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """

        if isinstance(ious_m, list):
            ious_m = np.asarray(ious_m, dtype=np.float32)
        ious_h50 = np.where(ious_m > nms_threshold)

        boxes = (
            predictions.pred_boxes[ious_h50] if predictions.has("pred_boxes") else None
        )
        scores = predictions.scores[ious_h50] if predictions.has("scores") else None
        classes = (
            predictions.pred_classes[ious_h50]
            if predictions.has("pred_classes")
            else None
        )

        keypoints = (
            predictions.pred_keypoints[ious_h50]
            if predictions.has("pred_keypoints")
            else None
        )

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks[ious_h50])
            masks = [
                GenericMask(x, self.output.height, self.output.width) for x in masks
            ]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            "thing_colors"
        ):
            colors = [[x / 255 for x in self.metadata.thing_colors[c]] for c in classes]
            alpha = 0.1

        else:
            colors = None
            alpha = 0.1

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert predictions.has(
                "pred_masks"
            ), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.1

        labels = self.create_text_labels_iou(
            classes,
            scores,
            ious_m[ious_h50] if len(ious_m) > 0 else np.array([]),
            self.metadata.get("thing_classes", None),
            noious,
            depths,
            volumes,
        )

        if obfuscate and not noious:
            labels = None
            alpha = 1.0
            colors = ["0.75"] * len(classes)  # grays
            boxes = None

        if noious:
            alpha = 0.1
            colors = colors
            boxes = None

        if add_labels is not None:
            labels = add_labels

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return self.output
