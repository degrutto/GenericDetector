### python preds_visualize.py --inputimgpath  /beegfs/gva.inaitcloud.com/projects/object_detection_results/ImagesForAnnotators_June2021/audia4_1.jpg --modeldir /beegfs/gva.inaitcloud.com/scratch/pipeline/v6/results/CarDetTrainAnonymize/spell/bassinet_titanium_revoke_flee_terrific/model0 --scorethresh 0.5
import argparse
from pathlib import Path
from axa.mrcnn.utils_det2infer2prodigy import build_predictor, build_prodigy_dict


def preds_visualizer_on_image(
    input_img_path: Path,
    model_dir: Path,
    score_threshold: float,
    nms_threshold: float = 0.5,
    obfuscate: bool = True,
    model_type: str = "DAMAGE_MASKS",
) -> list:
    """ main function to do anonymization of plate and faces using predictions from already trained model and then obfuscate the masks """

    predictor, metadata = build_predictor(
        model_dir, score_threshold, model_type, nms_threshold
    )

    payload = build_prodigy_dict(input_img_path, predictor, metadata)

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="set up inputs")

    parser.add_argument(
        "--inputimgpath",
        metavar="imagepath",
        required=True,
        help="the path of the img we want to visualize",
    )

    parser.add_argument(
        "--modeldir",
        metavar="trainedmodelpath",
        required=True,
        help="the path of the folder containing the model we pre-trained",
    )

    parser.add_argument(
        "--scorethresh",
        metavar="scorethreshold",
        required=True,
        help="the score threshold for inference",
    )

    args = parser.parse_args()
    preds_visualizer_on_image(
        input_img_path=args.inputimgpath,
        model_dir=args.modeldir,
        score_threshold=float(args.scorethresh),
    )
