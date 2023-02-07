from inait_prodigy.prodigy_config import MasksSpec, ProdigyConfig
from pydantic import ValidationError
import pytest
import os
import json


@pytest.fixture
def masks_spec():
    return MasksSpec(
        label_colors={
            "Plate": "#e53fe5",
            "Face": "#ff8c69",
            "Scratch": "#ffff00",
            "Broken": "#8b0000",
            "Crack": "#cc9933",
            "Dent": "#00ffff",
            "Gap": "#adff2f",
        }
    )


def test_prodigy_config(tmp_path, masks_spec):
    image_folder = os.path.join(tmp_path, "fake_images")
    dataset = "test_prodigy_config"
    os.makedirs(os.path.join(tmp_path, "fake_images"), exist_ok=True)
    for i in range(10):
        with open(os.path.join(image_folder, f"myfile_{i}.jpg"), "w") as fp:
            pass

    X = ProdigyConfig(
        image_folder=image_folder,
        dataset=dataset,
        redundancy=2,
        annotators=["a", "b", "c"],
        batches=[[f"myfile_{i}.jpg"] for i in range(10)],
        masks_spec=masks_spec,
    )

    with pytest.raises(ValidationError):
        ProdigyConfig(
            image_folder=image_folder,
            dataset=dataset,
            redundancy=2,
            annotators=["a", "b", "c"],
            batches=[[f"myfile_{i}.jpg"] for i in range(3)],
            masks_spec=masks_spec,
        )

    with pytest.raises(ValidationError):
        ProdigyConfig(
            image_folder=image_folder,
            dataset=dataset,
            redundancy=2,
            annotators=["a", "b", "c"],
            batches=[[f"myfile_{i}.jpg"] for i in range(3)]
            + [f"myfile_{i}.jpg" for i in range(3, 10)],
            masks_spec=masks_spec,
        )


def test_mask_spec_validation():
    with pytest.raises(ValidationError):
        MasksSpec(
            label_colors={
                "Plate": "#e53fe5",
                "Face": "#ff8c69",
                "Scratch": "#ffff00",
                "Broken": "#8b0000",
                "Crack": "#cc9933",
                "Bump": "#00ffff",
                "Gap": "#adff2f",
            }
        )

    with pytest.raises(ValidationError):
        MasksSpec(
            label_colors={
                "Plate": "#e53fe5",
                "Face": "#ff8c69",
                "Scratch": "#ffff00",
                "Broken": "#8b0000",
                "Crack": "#cc9933",
                "Dent": "#00ffff",
                "Gap": "#00ffff",
            }
        )

    with pytest.raises(ValidationError):
        MasksSpec(
            label_colors={
                "Plate": "#e53fe5",
                "Face": "#ff8c69",
                "Scratch": "#ffff00",
                "Broken": "#8b0000",
                "Crack": "#cc9933",
                "Dent": "#00ffff",
                "Gap": "toto",
            }
        )

    MasksSpec(
        label_colors={
            "Plate": "#e53fe5",
            "Face": "#ff8c69",
            "Scratch": "#ffff00",
            "Broken": "#8b0000",
            "Crack": "#cc9933",
            "Dent": "#00ffff",
            "Gap": "#adff2f",
        }
    )
