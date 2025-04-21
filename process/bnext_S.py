from PIL import Image
import onnxruntime as ort
import numpy as np
from pathlib import Path


# Trained on COCOFake dataset
class BNext_S_ModelONNX:
    def __init__(
        self, model_path="onnx_models/bnext_S_coco_model.onnx", resolution=224
    ):
        # Convert model_path to a Path object
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),  # Convert Path object to string for onnxruntime
        )
        self.resolution = resolution
        self.valid_extensions = (".jpg", ".jpeg", ".png")

    def apply_transforms(self, image):
        size = self.resolution + self.resolution // 8
        image = image.resize((size, size), resample=Image.BILINEAR)
        left = (size - self.resolution) // 2
        top = (size - self.resolution) // 2
        right = left + self.resolution
        bottom = top + self.resolution
        image = image.crop((left, top, right, bottom))
        arr = np.array(image).astype(np.float32) / 255.0
        if arr.ndim == 2:  # grayscale => add channel dimension
            arr = np.expand_dims(arr, axis=-1)
        arr = np.transpose(arr, (2, 0, 1))

        return arr[None, ...]

    def preprocess(self, image):
        return self.apply_transforms(image)

    def decode_prediction(self, confidence):

        confidence = confidence.item()

        label = (
            "likely fake"
            if confidence < 0.2
            else (
                "weakly fake"
                if confidence < 0.4
                else (
                    "uncertain"
                    if confidence < 0.6
                    else "weakly real" if confidence < 0.8 else "likely real"
                )
            )
        )

        return {"prediction": label, "confidence": confidence}

    def postprocess(self, output):
        logit = float(output[0][0])
        # numpy sigmoid
        prob = 1.0 / (1.0 + np.exp(-logit))
        return self.decode_prediction(prob)

    def predict(self, input):
        output = self.session.run(None, {"input": input})
        return output[0]
