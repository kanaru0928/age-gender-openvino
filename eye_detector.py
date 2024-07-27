import cv2


class EyeDetector:
    __ModelFile = (
        "model/intel/facial-landmarks-35-adas-0002/{}/facial-landmarks-35-adas-0002.xml"
    )

    __RightEyeInside = 0 << 1
    __RightEyeOutside = 1 << 1
    __LeftEyeInside = 2 << 1
    __LeftEyeOutside = 3 << 1

    def __init__(self, args, ie) -> None:
        self.ie = ie
        self.args = args
        self.model = ie.read_model(self.__ModelFile.format(args.model))
        self.compiled_model = ie.compile_model(self.model, device_name=args.device)
        self.input_layer = self.compiled_model.inputs
        self.output_layer = self.compiled_model.outputs

        self.fc = self.output_layer[0]

    def invoke(self, image):
        n, c, h, w = self.input_layer[0].shape
        input_image = cv2.resize(image, (w, h))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))

        result = self.compiled_model([input_image])
        landmarks = result[self.fc].squeeze()

        right_eye_inside = landmarks[self.__RightEyeInside : self.__RightEyeInside + 2]
        right_eye_outside = landmarks[
            self.__RightEyeOutside : self.__RightEyeOutside + 2
        ]
        left_eye_inside = landmarks[self.__LeftEyeInside : self.__LeftEyeInside + 2]
        left_eye_outside = landmarks[self.__LeftEyeOutside : self.__LeftEyeOutside + 2]
        
        right_eye = (right_eye_inside + right_eye_outside) / 2
        left_eye = (left_eye_inside + left_eye_outside) / 2

        return [right_eye, left_eye]

    def crop(self, image, eyes, margin_scale=0.15):
        image_h, image_w, _ = image.shape

        cropped_eyes = []

        for eye in eyes:
            x, y = eye
            x = int(x * image_w)
            y = int(y * image_h)

            margin_x = int(image_w * margin_scale)
            margin_y = int(image_h * margin_scale)

            x_min = max(0, x - margin_x)
            y_min = max(0, y - margin_y)
            x_max = min(image_w, x + margin_x)
            y_max = min(image_h, y + margin_y)

            cropped_eye = image[y_min:y_max, x_min:x_max]
            cropped_eyes.append(cropped_eye)

        return cropped_eyes

    def visualize(
        self,
        image,
        face,
        eyes,
        thickness=3,
        margin_scale=0.15,
        cropped_scale=0.2,
    ):
        image_h, image_w, _ = image.shape

        face_x_min = int(face[3] * image_w)
        face_y_min = int(face[4] * image_h)
        face_x_max = int(face[5] * image_w)
        face_y_max = int(face[6] * image_h)

        face_margin_x = int((face_x_max - face_x_min) * cropped_scale)
        face_margin_y = int((face_y_max - face_y_min) * cropped_scale)

        face_x_min = max(0, face_x_min - face_margin_x)
        face_y_min = max(0, face_y_min - face_margin_y)
        face_x_max = min(image_w, face_x_max + face_margin_x)
        face_y_max = min(image_h, face_y_max + face_margin_y)

        face_w = face_x_max - face_x_min
        face_h = face_y_max - face_y_min

        for eye in eyes:
            x, y = eye
            x = int(x * face_w)
            y = int(y * face_h)

            margin = int(face_w * margin_scale)

            x_min = max(0, x - margin) + face_x_min
            y_min = max(0, y - margin) + face_y_min
            x_max = min(image_w, x + margin) + face_x_min
            y_max = min(image_h, y + margin) + face_y_min

            cv2.rectangle(
                image, (x_min, y_min), (x_max, y_max), (255, 255, 255), thickness
            )

        return image
