# imports
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
from process.bnext_m import BNext_M_ModelONNX
from process.bnext_s import BNext_S_ModelONNX
from process.transformer import TransformerModelONNX
from random import randint
import os
from sim_data import defaultDataset

warnings.filterwarnings("ignore")


# Configure UI Elements in RescueBox Desktop
def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing all images",
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

models = [BNext_M_ModelONNX(), BNext_S_ModelONNX(), TransformerModelONNX()]


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

            # Preprocess, predict, postprocess
            preprocessed_image = model.preprocess(image)
            prediction = model.predict(preprocessed_image)
            processed_prediction = model.postprocess(prediction)

            # Add image name to prediction
            processed_prediction["image_path"] = image_path

            # Append the result to the list
            model_results.append(processed_prediction)

        results.append(model_results)
    return results


@server.route(
    "/predict",
    task_schema_func=create_transform_case_task_schema,
    short_title="DeepFake Detection",
    order=0,
)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path
    out = Path(inputs["output_file"].path)
    # Need logic to verify that the random num is not already in the directory *******
    random_num = randint(0, 999)
    out = out / (f"predictions_{random_num}.csv")

    dataset = defaultDataset(dataset_path=input_path, resolution=224)

    res_list = run_models(models, dataset)
    # Prepare model data structure
    model_data = []
    for model_results in res_list:
        model_name = model_results[0]["model_name"]
        predictions = model_results[1:]
        model_data.append({"name": model_name, "predictions": predictions})

    # Build CSV content
    csv_rows = []
    # Add model names header
    csv_rows.append(["Model:"] + [m["name"] for m in model_data])

    # Add prediction rows grouped by path
    for i in range(len(model_data[0]["predictions"])):
        # Path row
        path = [os.path.basename(model_data[0]["predictions"][i]["image_path"])] * len(
            models
        )

        # Prediction row
        preds = [m["predictions"][i]["prediction"] for m in model_data]

        # Confidence row (as percentages)
        confidences = [
            f"{m['predictions'][i]['confidence'] * 100:.0f}%" for m in model_data
        ]

        # Add the rows
        csv_rows.append(["Path:"] + path)
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
    server.run(port=args.port)
