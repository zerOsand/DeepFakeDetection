import torchvision.transforms.v2 as T
from PIL import Image
import torch
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
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )(image)[
            None,
        ].numpy()

    def preprocess(self, image):
        return self.apply_transforms(image)

    def decode_prediction(self, confidence):

        confidence = confidence.item()

        label = ("likely fake" if confidence < 0.2 else
                 "weakly fake" if confidence < 0.4 else
                 "uncertain" if confidence < 0.6 else
                 "weakly real" if confidence < 0.8 else
                 "likely real")
        
        return {"prediction": label, "confidence": confidence}

    def postprocess(self, output):
        return self.decode_prediction(torch.sigmoid(torch.tensor(output[0][0].item())))

    def predict(self, input):
        output = self.session.run(None, {"input": input})
        return output[0]
