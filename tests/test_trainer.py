import pytest
from scripts.trainer import train

@pytest.mark.run_on_gpu
def test_instance_trainer():
    #instance_trainer_inputs = instance_trainer_inputs()
    train(
        jsonl_file = "tests/assets_objdet/annotations_reduced.jsonl",
        imgs_base_path= "tests/assets_objdet/",
        modeltype = "instance_segmentation",
        classes =  ["Scratch",  "Broken"] ,
        modelinitpath=None
    )


if __name__ == "__main__":
    test_insrance_trainer()
