# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained(
    "dima806/deepfake_vs_real_image_detection"
)
model = AutoModelForImageClassification.from_pretrained(
    "dima806/deepfake_vs_real_image_detection"
)
