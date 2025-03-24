import argparse

# import matplotlib.pyplot as plt
# from BNN github
# import model as model
# import numpy as np

from sim_data import defaultDataset

from process.transformer_process import TransformerModel
from process.bnext_process import BNextModelONNX


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/test",
        help="Path to the dataset folder.",
    )
    # TODO add arguments for models to use
    args = parser.parse_args()
    return args


def get_area_ratio(img1, img2):
    print(img1, img2)
    h1, w1 = img1
    h2, w2 = img2
    return (h1 * w1) / (h2 * w2)


# Inputs: models (list of model objects), dataset (dataset object)
# Outputs: results (a list of lists of dictionaries, one for each model)
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


if __name__ == "__main__":

    # print(torch.cuda.is_available())
    # exit()
    input = "sample_input/real"

    args = args_func()
    # input = args.dataset_path

    test_dataset = defaultDataset(dataset_path=input, resolution=224)

    models_to_use = [BNextModelONNX("onnx_models/bnext_model.onnx"), TransformerModel()]

    results = run_models(models_to_use, test_dataset)

    print(results)
