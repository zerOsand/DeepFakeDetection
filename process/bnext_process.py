import torchvision.transforms.v2 as T
from PIL import Image
import torch
import os
import onnxruntime as ort
import numpy as np


class BNextModelONNX:
    def __init__(self, model_path="onnx_models/bnext_model.onnx", resolution=224):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.resolution = resolution
        self.valid_extensions = (".jpg", ".jpeg", ".png")

    def apply_transforms(self, image):
        return T.Compose(
            [
                T.Resize(
                    self.resolution + self.resolution // 8,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
            ]
        )(image)[
            None,
        ].numpy()

    def preprocess(self, image):
        return self.apply_transforms(image)

    def decode_prediction(self, prediction):
        conf = prediction if prediction > 0.5 else 1 - prediction
        pred = "real" if prediction > 0.8 else "fake" if conf > 0.8 else "uncertain"
        return {"prediction": pred, "confidence": conf.item()}

    def postprocess(self, output):
        return self.decode_prediction(torch.sigmoid(torch.tensor(output[0][0].item())))

    def predict(self, input):
        output = self.session.run(None, {"input": input})
        return output[0]
