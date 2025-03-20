from os import listdir
from os.path import isdir

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import Dataset


class defaultDataset(Dataset):
    def __init__(self, dataset_path, resolution=224):
        assert isdir(dataset_path), f"Dataset path {dataset_path} does not exist."
        self.dataset_path = dataset_path
        self.resolution = resolution

        self.images = []
        for image_path in listdir(dataset_path):
            if (
                image_path.lower().endswith(".jpg")
                or image_path.lower().endswith(".png")
                or image_path.lower().endswith(".jpeg")
            ):
                self.images.append(
                    dataset_path
                    + "/"
                    + image_path
                    + ("F" if image_path[0] == "F" else "R")
                )

    def __len__(self):
        return len(self.images)

    def read_image(self, path):
        image = Image.open(path).convert("RGB")
        original_res = image.size
        new_image = T.Compose(
            [
                T.Resize(
                    self.resolution + self.resolution // 8,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
            ]
        )(image)
        return new_image, original_res, image

    def apply_transforms(self, image):
        image = Image.fromarray(image)
        return T.Compose(
            [
                T.Resize(
                    self.resolution + self.resolution // 8,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.CenterCrop(self.resolution),
                T.ToTensor(),
            ]
        )(image)

    def __getitem__(self, i):
        try:
            image, res, raw = self.read_image(self.images[i][:-1])
        except:
            print(f"Error reading image {self.images[i]}")
            return None
        sample = {
            "image_path": self.images[i],
            "image": image,
            "is_real": torch.tensor([1 if self.images[i][-1] == "R" else 0]),
            "original_res": res,
            "raw": raw,
        }
        return sample
