from openvino.runtime import Core
import cv2
import numpy as np
from logger import get_logger


logger = get_logger(__name__)


class FaceDetector:
    __ModelFile = "model/intel/face-detection-adas-0001/{}/face-detection-adas-0001.xml"
    __MaxWidth = 640

    def __init__(self, args, ie: Core) -> None:
        self.ie = ie
        self.args = args
        self.model = ie.read_model(self.__ModelFile.format(args.model))
        self.compiled_model = ie.compile_model(self.model, device_name=args.device)
        self.input_layer = self.compiled_model.inputs
        self.output_layer = self.compiled_model.outputs

    def invoke(self, image):
        scale = self.__MaxWidth / image.shape[1]
        image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
        image_h, image_w, _ = image.shape
        logger.info(f"image shape: {image.shape}")

        n, c, h, w = self.input_layer[0].shape
        input_image = cv2.resize(image, (w, h))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))

        result = self.compiled_model([input_image])[self.output_layer[0]]
        logger.info(f"result shape: {result.shape}")

        faces = result[0][0][np.where(result[0][0][:, 2] > 0.5)]

        return faces

    def crop(self, image, faces, margin_scale=0):
        image_h, image_w, _ = image.shape

        cropped_faces = []
        new_faces = []
        
        for face in faces:
            x_min = int(face[3] * image_w)
            y_min = int(face[4] * image_h)
            x_max = int(face[5] * image_w)
            y_max = int(face[6] * image_h)

            margin_x = int((x_max - x_min) * margin_scale)
            margin_y = int((y_max - y_min) * margin_scale)

            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(image_w, x_max + margin_x)
            y_max = min(image_h, y_max + margin_y)

            cropped_face = image[y_min:y_max, x_min:x_max]
            cropped_faces.append(cropped_face)
            
            new_faces.append([x_min, y_min, x_max, y_max])
            
        return cropped_faces, new_faces

    def visualize(
        self, image, faces, thickness=2, font_scale=0.9, texts=None
    ):
        image_h, image_w, _ = image.shape
        if texts is None:
            texts = [f"face {i}" for i in range(len(faces))]

        for i, face in enumerate(faces):

            x_min = int(face[3] * image_w)
            y_min = int(face[4] * image_h)
            x_max = int(face[5] * image_w)
            y_max = int(face[6] * image_h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness)
            cv2.putText(
                image,
                texts[i],
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (36, 255, 12),
                thickness,
            )

        return image
