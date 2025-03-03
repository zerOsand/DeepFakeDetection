import argparse

import matplotlib.pyplot as plt
# from BNN github
# import model as model
# import numpy as np
import torch
import tqdm
from retinaface import RetinaFace
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
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="weights/dffd_M_unfrozen.ckpt",
        help="Path to the model checkpoint.",
    )
    args = parser.parse_args()
    return args


def get_area_ratio(img1, img2):
    print(img1, img2)
    h1, w1 = img1
    h2, w2 = img2
    return (h1 * w1) / (h2 * w2)


if __name__ == "__main__":

    # print(torch.cuda.is_available())
    # exit()
    cfg = {
        "dataset_path": "datasets/test",
        "resolution": 224,
        "ckpt": "weights/dffd_M_unfrozen.ckpt",
        "enable_facecrop": True,
    }

    args = args_func()

    cfg["dataset_path"] = args.dataset_path
    cfg["ckpt"] = args.ckpt_path

    test_dataset = defaultDataset(
        dataset_path=cfg["dataset_path"], resolution=cfg["resolution"]
    )

    model_ckpt = cfg["ckpt"]
    net = model.BNext4DFR.load_from_checkpoint(model_ckpt)
    print(f"Model loaded from {model_ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    net = net.to(device)

    net.eval()
    label = []
    logits = []
    tp = tn = fp = fn = 0
    err_imgs = 0
    uncertain_imgs = 0
    # ret_model =
    for i in tqdm.tqdm(range(len(test_dataset))):
        sample = test_dataset[i]
        if sample is None:
            err_imgs += 1
            continue
        image = sample["image"]
        if cfg["enable_facecrop"]:
            faces = RetinaFace.extract_faces(
                sample["image_path"][:-1], expand_face_area=100, align=False
            )
            if len(faces) > 0:
                image = test_dataset.apply_transforms(faces[0])
            else:
                image = sample["image"]

        image = torch.tensor(image)
        image = image.unsqueeze(0).to(device)
        is_real = sample["is_real"].to(device)
        with torch.no_grad():
            output = net(image)
            logit = output["logits"][0][0]
            temp = torch.sigmoid(logit)
            if temp > 0.2 and temp < 0.8:
                uncertain_imgs += 1
            logits.append(logit)
            label.append(is_real[0])

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