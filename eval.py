import argparse

import matplotlib.pyplot as plt
# from BNN github
# import model as model
# import numpy as np
import torch
import tqdm
# from retinaface import RetinaFace
from torchmetrics.functional.classification import (accuracy, confusion_matrix,
                                                    f1_score)

from sim_data import defaultDataset


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/test",
        help="Path to the dataset folder.",
    )
    #TODO add arguments for models to use
    args = parser.parse_args()
    return args


def get_area_ratio(img1, img2):
    print(img1, img2)
    h1, w1 = img1
    h2, w2 = img2
    return (h1 * w1) / (h2 * w2)


#Inputs: models (list of model objects), dataset (dataset object)
def run_models(models, dataset):


if __name__ == "__main__":

    # print(torch.cuda.is_available())
    # exit()
    input = "sample_input/real"

    args = args_func()
    input = args.dataset_path

    test_dataset = defaultDataset(
        dataset_path=input, resolution=224
    )

    

    preds = torch.stack(logits)
    target = torch.stack(label)
    acc = accuracy(preds, target, task="binary", average="micro", threshold=0.5)
    f1 = f1_score(preds, target, task="binary", threshold=0.5, average="micro")
    cm = confusion_matrix(preds, target, task="binary", threshold=0.5)
    print(f"F1: {f1}")
    print(f"Accuracy: {acc}")
    print(cm)
    print(f"Uncertain images: {uncertain_imgs}")
    print(f"Error images: {err_imgs}")