from PIL import Image
import onnxruntime as ort
import numpy as np
from pathlib import Path
import process.facedetector as facedetector
from process.utils import (
    Compose,
    InterpolationMode,
    Resize,
    CenterCrop,
    ToImage,
    ToDtype,
)


# Trained on COCOFake dataset
class BNext_M_ModelONNX:
    def __init__(
        self, model_path="onnx_models/bnext_M_dffd_model.onnx", resolution=224
    ):
        # Convert model_path to a Path object
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path),  # Convert Path object to string for onnxruntime
        )
        self.resolution = resolution
        self.valid_extensions = (".jpg", ".jpeg", ".png")

    def apply_transforms(self, image: Image.Image) -> np.ndarray:
        transform = Compose(
            [
                Resize(
                    self.resolution + self.resolution // 8,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                CenterCrop(self.resolution),
                ToImage(),
                ToDtype(np.float32, scale=True),
            ]
        )
        out = transform(image)  # H×W×C float32 in [0,1]
        out = out.transpose(2, 0, 1)
        return out[None, ...]  # add batch dim

    def preprocess(self, image, facecrop=None):
        if facecrop:
            self.resolution_ratio = 1.5  # Default value if not set
            # Assuming faceDetector returns (center_x, center_y) or None
            # The faceDetector might need a specific input format (e.g., numpy array)
            face_center_coords = None
            try:
                # Convert PIL Image to numpy array (RGB)
                np_image = np.array(image.convert('RGB'))

                # Assuming faceDetector takes a numpy array and the ONNX session
                face_center_coords = facedetector.faceDetector(np_image, face_detector=facecrop)[3]
            except Exception as e:
                # Handle potential errors during face detection
                print(f"Warning: Face detection failed with error: {e}. Proceeding without cropping.")


            if face_center_coords is not None:
                # Define the ratio for cropping relative to the base resolution.
                if not hasattr(self, 'resolution_ratio'):
                    print("Warning: self.resolution_ratio not set. Defaulting to 1.0.")
                    self.resolution_ratio = 1.0 # Default value if not set

                center_x, center_y = face_center_coords
                img_width, img_height = image.size
                # Desired crop size based on resolution and ratio, centered around the face
                crop_half_size = int(self.resolution * self.resolution_ratio / 2) # Half the desired dimension

                # Calculate initial crop box coordinates
                left = center_x - crop_half_size
                top = center_y - crop_half_size
                right = center_x + crop_half_size
                bottom = center_y + crop_half_size

                # Clamp coordinates to be within image boundaries
                left = max(0, left)
                top = max(0, top)
                right = min(img_width, right)
                bottom = min(img_height, bottom)

                # Ensure coordinates are integers for PIL crop
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)

                # Check if the calculated box has valid dimensions (width > 0 and height > 0)
                if right > left and bottom > top:
                    # Crop the original PIL image
                    image = image.crop((left, top, right, bottom))
                    # Optional: Save the cropped image for debugging
                    # Path("temp").mkdir(exist_ok=True) # Ensure temp directory exists
                    # image.save(f"temp/cropped_{center_x}_{center_y}.jpg")
                else:
                    # Optional: Log or print a warning if the crop box is invalid
                    print(f"Warning: Invalid crop box calculated ({left}, {top}, {right}, {bottom}) for face at ({center_x}, {center_y}). Using original image.")

        # Apply standard transforms to the (potentially cropped) image
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
