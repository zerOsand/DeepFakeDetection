import argparse
import csv
import warnings
from typing import TypedDict
from pathlib import Path
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)
from process.bnext_process import BNextModelONNX
from process.transformer_process import TransformerModel
import torch

from sim_data import defaultDataset

warnings.filterwarnings("ignore")


# Configure UI Elements in RescueBox Desktop
def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output file",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])


# Specify the input and output types for the task
class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput


class Parameters(TypedDict):
    pass


# Create a server instance
server = MLServer(__name__)

server.add_app_metadata(
    name="Image DeepFake Detector",
    author="UMass Rescue",
    version="0.2.0",
    info=load_file_as_string("img-app-info.md"),
)

models = [BNextModelONNX("onnx_models/bnext_model.onnx"), TransformerModel()]
# model = DeepFakeModel("deepfake_image_model.onnx")

def run_models(models, dataset):
    results = []
    for model in models:
        model_results = []
        model_results.append({"model_name": model.__class__.__name__})
        for i in range(
            len(dataset)
        ):  # This is done one image at a time to avoid memory issues
            sample = dataset[i]
            image = sample["image"]
            image_path = sample["image_path"]
            original_res = sample["original_res"]

            # Preprocess the image
            preprocessed_image = model.preprocess(image)

            # Get the prediction
            prediction = model.predict(preprocessed_image)

            # Postprocess the prediction
            processed_prediction = model.postprocess(prediction)

            # Add the name of the image to the prediction
            processed_prediction["image_path"] = image_path

            # Append the result to the list
            model_results.append(processed_prediction)

        results.append(model_results)

    return results

@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path
    out = Path(inputs["output_file"].path)
    out = str(out / (f"predictions_" + str(int(torch.rand(1) * 1000)) + ".csv"))

    dataset = defaultDataset(dataset_path=input_path, resolution=224)

    # print(parameters)
    res_list = run_models(models, dataset)
    # print(res_list)
    # Prepare model data structure
    model_data = []
    for model_results in res_list:
        model_name = model_results[0]["model_name"]
        predictions = model_results[1:]
        model_data.append({
            "name": model_name,
            "predictions": predictions
        })
    
    # Build CSV content
    csv_rows = []
    # Add model names header
    csv_rows.append(["Model:"] + [m["name"] for m in model_data])
    
    # Add prediction rows grouped by path
    for i in range(len(model_data[0]["predictions"])):
        # Path row
        paths = [m["predictions"][i]["image_path"] for m in model_data]

        # Prediction row
        preds = [m["predictions"][i]["prediction"] for m in model_data]
        
        # Confidence row (as percentages)
        confidences = [f"{m['predictions'][i]['confidence'] * 100:.0f}%" 
                      for m in model_data]

        # Add the rows
        csv_rows.append(["Path:"] + paths)
        csv_rows.append(["Prediction:"] + preds)
        csv_rows.append(["Confidence:"] + confidences)
    
    # Write to CSV
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    return ResponseBody(FileResponse(path=str(out), file_type="csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()
    print(
        "CUDA is available." if torch.cuda.is_available() else "CUDA is not available."
    )
    server.run(port=args.port)
