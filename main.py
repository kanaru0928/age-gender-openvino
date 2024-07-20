import cv2
from openvino.runtime import Core
import argparse

from age_gender_predictor import AgeGenderPredictor
from face_detector import FaceDetector
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

    detector = FaceDetector(args, ie)
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.invoke(image)
    cropped_faces = detector.crop(image, faces, margin_scale=0.2)

    age_gender_predictor = AgeGenderPredictor(args, ie)

    age_list = []
    gender_list = []

    for cropped_face in cropped_faces:
        age, gender = age_gender_predictor.invoke(cropped_face)
        age_list.append(age)
        gender_list.append(gender)

    texts = [f"{gender}:{age:.1f}" for gender, age in zip(gender_list, age_list)]

    detector.visualize(image, faces, thickness=3, font_scale=1.2, texts=texts)


if __name__ == "__main__":
    args = parse()
    main(args)
