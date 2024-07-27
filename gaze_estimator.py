import cv2
import numpy as np


class GazeEstimator:
    __ModelFile = (
        "model/intel/gaze-estimation-adas-0002/{}/gaze-estimation-adas-0002.xml"
    )

    def __init__(self, args, ie) -> None:
        self.ie = ie
        self.args = args
        self.model = ie.read_model(self.__ModelFile.format(args.model))
        self.compiled_model = ie.compile_model(self.model, device_name=args.device)
        self.input_layer = self.compiled_model.inputs
        self.output_layer = self.compiled_model.outputs

        self.fc = self.output_layer[0]

    def invoke(self, left_eye_image, right_eye_image, headpose):
        n, c, h, w = self.input_layer[0].shape

        input_left_eye_image = cv2.resize(left_eye_image, (w, h))
        input_left_eye_image = input_left_eye_image.transpose((2, 0, 1))
        input_left_eye_image = input_left_eye_image.reshape((n, c, h, w))

        input_right_eye_image = cv2.resize(right_eye_image, (w, h))
        input_right_eye_image = input_right_eye_image.transpose((2, 0, 1))
        input_right_eye_image = input_right_eye_image.reshape((n, c, h, w))

        input_headpose = np.array(headpose).reshape((1, 3))

        result = self.compiled_model(
            [input_left_eye_image, input_right_eye_image, input_headpose]
        )
        gaze = result[self.fc].squeeze()

        return gaze

    def visualize(self, image, face, gaze, thickness=2, cropped_scale=0.15, length=150):
        image_h, image_w, _ = image.shape

        x_min = int(face[3] * image_w)
        y_min = int(face[4] * image_h)
        x_max = int(face[5] * image_w)
        y_max = int(face[6] * image_h)

        margin_x = int((x_max - x_min) * cropped_scale)
        margin_y = int((y_max - y_min) * cropped_scale)

        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(image_w, x_max + margin_x)
        y_max = min(image_h, y_max + margin_y)

        cv2.arrowedLine(
            image,
            (int((x_min + x_max) / 2), int((y_min + y_max) / 2)),
            (
                int((x_min + x_max) / 2 + gaze[0] * length),
                int((y_min + y_max) / 2 - gaze[1] * length),
            ),
            (0, 0, 255),
            thickness,
            line_type=cv2.LINE_AA,
            tipLength=0.2
        )

        return image
