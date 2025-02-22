import torchvision.transforms.v2 as T
from PIL import Image
import torch
import os
import onnxruntime as ort
import numpy as np


class DeepFakeProcessing:
    def __init__(self, resolution=224):
        self.resolution = resolution
        self.valid_extensions = (".jpg", ".jpeg", ".png")

    def apply_transforms_on_path(self, path):
        image = Image.open(path).convert("RGB")
        return self.apply_transforms(image)

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

    def find_images_in_dir(self, directory):
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(self.valid_extensions)
        ]

    def preprocess(self, dir_path):
        return [
            self.preprocess_single(path) for path in self.find_images_in_dir(dir_path)
        ]

    def preprocess_single(self, image_path):
        return self.apply_transforms_on_path(image_path)

    def decode_prediction(self, prediction):
        conf = prediction if prediction > 0.5 else 1 - prediction
        pred = "real" if prediction > 0.8 else "fake" if conf > 0.8 else "uncertain"
        return {"prediction": pred, "confidence": conf.item()}

    def postprocess(self, outputs):
        return [self.postprocess_single(out) for out in outputs]

    def postprocess_single(self, output):
        return self.decode_prediction(torch.sigmoid(torch.tensor(output[0][0].item())))


class DeepFakeModel:
    def __init__(self, model_path):
        self.dfp = DeepFakeProcessing()
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(self, image_path):
        input = self.dfp.preprocess_single(image_path)
        output = self.session.run(None, {"input": input})
        out = self.dfp.postprocess_single(output[0])
        out["image_path"] = image_path
        return out

    def predict_dir(self, input_dir):
        outputs = []
        for inp in self.dfp.find_images_in_dir(input_dir):
            outputs.append(self.predict(inp))
        return outputs
