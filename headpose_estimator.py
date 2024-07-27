import cv2


class HeadposeEstimator:
    __ModelFile = (
        "model/intel/head-pose-estimation-adas-0001/{}/head-pose-estimation-adas-0001.xml"
    )

    def __init__(self, args, ie) -> None:
        self.ie = ie
        self.args = args
        self.model = ie.read_model(self.__ModelFile.format(args.model))
        self.compiled_model = ie.compile_model(self.model, device_name=args.device)
        self.input_layer = self.compiled_model.inputs
        self.output_layer = self.compiled_model.outputs

        self.fc_r = self.output_layer[0]
        self.fc_p = self.output_layer[1]
        self.fc_y = self.output_layer[2]

    def invoke(self, image):
        n, c, h, w = self.input_layer[0].shape
        input_image = cv2.resize(image, (w, h))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w))

        result = self.compiled_model([input_image])
        headpose = (
            result[self.fc_r].squeeze(),
            result[self.fc_p].squeeze(),
            result[self.fc_y].squeeze(),
        )

        return headpose
