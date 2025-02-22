import onnxruntime as ort
import numpy as np
from onnx_helper import DeepFakeModel
import argparse
from pathlib import Path
import json
from pprint import pprint

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


model = DeepFakeModel("deepfake_image_model.onnx")
outputs = model.predict_dir(input_dir)
pprint(outputs)
with open(output_dir / "output.json", "w") as f:
    json.dump(outputs, f)
