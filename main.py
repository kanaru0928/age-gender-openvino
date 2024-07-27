import cv2
from openvino.runtime import Core
import argparse
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from age_gender_predictor import AgeGenderPredictor
from eye_detector import EyeDetector
from face_detector import FaceDetector
from gaze_estimator import GazeEstimator
from headpose_estimator import HeadposeEstimator
from logger import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


def parse():
    parser = argparse.ArgumentParser(description="Face Detection")
    parser.add_argument("--device", type=str, default="CPU", help="Device")
    parser.add_argument("--model", type=str, default="FP16", help="Model")
    parser.add_argument("--image", type=str, required=True, help="Image")
    args = parser.parse_args()
    return args


def main(args):
    ie = Core()

    detector = FaceDetector(args, ie)
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.invoke(image)
    cropped_faces, new_faces = detector.crop(image, faces, margin_scale=0.2)

    age_gender_predictor = AgeGenderPredictor(args, ie)

    age_list = []
    gender_list = []

    for cropped_face in cropped_faces:
        age, gender = age_gender_predictor.invoke(cropped_face)
        age_list.append(age)
        gender_list.append(gender)

    texts = [f"{gender}:{age:.1f}" for gender, age in zip(gender_list, age_list)]

    image = detector.visualize(image, faces, thickness=3, font_scale=1.2, texts=texts)

    for face in new_faces:
        x_min, y_min, x_max, y_max = face
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

    eye_detector = EyeDetector(args, ie)

    cropped_eyes = []
    for cropped_face, face in zip(cropped_faces, faces):
        eyes = eye_detector.invoke(cropped_face)
        image = eye_detector.visualize(
            image, face, eyes, thickness=2, margin_scale=0.15
        )
        cropped_eyes.append(eye_detector.crop(cropped_face, eyes, margin_scale=0.15))

    headpose_estimator = HeadposeEstimator(args, ie)
    headpose_list = []

    for cropped_face in cropped_faces:
        headpose = headpose_estimator.invoke(cropped_face)
        headpose_list.append(headpose)

    gaze_estimator = GazeEstimator(args, ie)

    for cropped_eye, face in zip(cropped_eyes, faces):
        gaze = gaze_estimator.invoke(cropped_eye[0], cropped_eye[1], headpose)
        image = gaze_estimator.visualize(
            image, face, gaze, thickness=2, cropped_scale=0.15
        )

    image_h, image_w, _ = image.shape

    file_name = (
        "output/"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
        f"_{args.image.split('/')[-1]}"
    )

    plt.figure(figsize=(image_w / 100, image_h / 100), dpi=100)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(image)
    plt.savefig(file_name)


if __name__ == "__main__":
    args = parse()
    main(args)
