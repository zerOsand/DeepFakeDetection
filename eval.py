import argparse
from sim_data import defaultDataset
from process.transformer import TransformerModelONNX
from process.bnext import BNextModelONNX
import os
import json
import pandas as pd

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset folder that contains the test images. The root directory is where this file is located.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",  # Accepts one or more model names as a list
        required=True,
        help="List of models to use (e.g., TransformerModel BNextModelONNX). Use 'all' to run all models or 'list' to list available models.",
    )

    args = parser.parse_args()
    return args

# def get_area_ratio(img1, img2):
#     print(img1, img2)
#     h1, w1 = img1
#     h2, w2 = img2
#     return (h1 * w1) / (h2 * w2)


# Inputs: models (list of model objects), dataset (dataset object)
# Outputs: results (a list of lists of dictionaries, one for each model)
def run_models(models, dataset):
    results = []
    for model in models:
        print(f"Running model: {model.__class__.__name__}")
        model_results = []
        model_results.append({"model_name": model.__class__.__name__})
        for i in range(
            len(dataset)
        ):  # This is done one image at a time to avoid memory issues
            sample = dataset[i]
            print(sample)
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


if __name__ == "__main__":
    available_models = {
         cls.__name__: cls
         for cls in [BNextModelONNX, TransformerModelONNX]
     }
    input_path = "sample_input"

    args = args_func()
    if args.dataset_path:
        input_path = args.dataset_path
    # Check if the input is a valid path
    if not os.path.exists(input_path):
        raise ValueError(f"Invalid path: {input_path}")

    models_to_use = []
    for model in args.models:
        if model == "all":
            models_to_use = [cls() for cls in available_models.values()]
            print("Using all models")
            break
        elif model == "list":
            print("Available models:")
            for model_name in available_models.keys():
                print(model_name)
            exit()
        elif model in available_models:
            models_to_use.append(available_models[model]())
        else:
            # print that model: model is not available
            print(f"Model: {model} is not available")
    if len(models_to_use) == 0:
        raise ValueError(
            "No valid models were selected. Please select at least one model, use 'list' to see available models, or 'all' to use all models."
        )
    else:
        print("Using models:")
        for model in models_to_use:
            print(model.__class__.__name__)
        print("--" * 20)
        print("Using dataset:", input_path)
        print("--" * 20)
        print("Proceed? (y/n)")
        proceed = input().strip().lower()
        if proceed != "y":
            print("Exiting...")
            exit()

    test_dataset = defaultDataset(dataset_path=input_path, resolution=224)

    results = run_models(models_to_use, test_dataset)

    with open("sample_output/out.json", "w") as f:
        json.dump(results, f, indent=4)
    
    flattened_results = [item for model_results in results for item in model_results]
    o = pd.DataFrame(flattened_results)  # Convert to a pandas DataFrame
    o.to_csv("sample_output/out.csv", index=False)  # Save as a CSV file