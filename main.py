import cv2
from openvino.runtime import Core
import matplotlib.pyplot as plt
import numpy as np
import argparse

from logger import get_logger

logger = get_logger(__name__)


def parse():
    parser = argparse.ArgumentParser(description="Face Detection")
    parser.add_argument("--device", type=str, default="CPU", help="Device")
    parser.add_argument("--model", type=str, default="FP16", help="Model")
    parser.add_argument("--image", type=str, default="yobinori.png", help="Image")
    args = parser.parse_args()
    return args


def main(args):
    ie = Core()
    crop_model = ie.read_model(
        f"model/intel/face-detection-adas-0001/{args.model}/face-detection-adas-0001.xml"
    )
    compiled_crop_model = ie.compile_model(crop_model, device_name=args.device)
    input_layer = compiled_crop_model.inputs
    output_layer = compiled_crop_model.outputs

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale = 640 / image.shape[1]
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    image_h, image_w, _ = image.shape
    logger.info(f"image shape: {image.shape}")

    n, c, h, w = input_layer[0].shape
    input_image = cv2.resize(image, (w, h))
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image.reshape((n, c, h, w))

    result = compiled_crop_model([input_image])[output_layer[0]]
    logger.info(f"result shape: {result.shape}")

    faces = result[0][0][np.where(result[0][0][:, 2] > 0.5)]

    for i, face in enumerate(faces):
        x_min = int(face[3] * image_w)
        y_min = int(face[4] * image_h)
        x_max = int(face[5] * image_w)
        y_max = int(face[6] * image_h)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            image,
            str(i),
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    args = parse()
    main(args)
