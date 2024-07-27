import cv2
import numpy as np

from logger import get_logger

logger = get_logger(__name__)

class AgeGenderPredictor:
    __ModelFile = (
        "model/intel/age-gender-recognition-retail-0013/{}/"
        "age-gender-recognition-retail-0013.xml"
    )
    __Label = ["Female", "Male"]

    def __init__(self, args, ie):
        self.ie = ie
        self.args = args
        self.model = ie.read_model(self.__ModelFile.format(args.model))
        self.compiled_model = ie.compile_model(self.model, device_name=args.device)
        self.input_layer = self.compiled_model.inputs
        self.output_layer = self.compiled_model.outputs

        self.prob = self.output_layer[0]
        self.age_out = self.output_layer[1]

    def invoke(self, image):
        n, c, h, w = self.input_layer[0].shape
        input_image = cv2.resize(image, (w, h))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))

        result = self.compiled_model([input_image])

        gender = self.__Label[np.argmax(result[self.prob].squeeze())]
        
        male_prob = result[self.prob].squeeze()[1]
        if male_prob > 0.5 and male_prob < 0.85:
            gender += "?"
        
        age = result[self.age_out].squeeze() * 100

        return age, gender
