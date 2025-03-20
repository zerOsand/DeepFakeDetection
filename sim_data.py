from os import listdir
from os.path import isdir

import numpy as np
from PIL import Image


class defaultDataset:
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
        
        # Resize and crop without torchvision
        target_size = self.resolution + self.resolution // 8
        image = image.resize((target_size, target_size), Image.BILINEAR)
        
        # Center crop
        width, height = image.size
        left = (width - self.resolution) // 2
        top = (height - self.resolution) // 2
        right = left + self.resolution
        bottom = top + self.resolution
        image = image.crop((left, top, right, bottom))
        
        # Convert to numpy array and normalize to [0,1]
        new_image = np.array(image).astype(np.float32) / 255.0
        # Reshape to match the expected format (channels first)
        new_image = np.transpose(new_image, (2, 0, 1))
        
        return new_image, original_res, image

    def apply_transforms(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Resize and crop
        target_size = self.resolution + self.resolution // 8
        image = image.resize((target_size, target_size), Image.BILINEAR)
        
        # Center crop
        width, height = image.size
        left = (width - self.resolution) // 2
        top = (height - self.resolution) // 2
        right = left + self.resolution
        bottom = top + self.resolution
        image = image.crop((left, top, right, bottom))
        
        # Convert to numpy array and normalize to [0,1]
        image_array = np.array(image).astype(np.float32) / 255.0
        # Reshape to match the expected format (channels first)
        image_array = np.transpose(image_array, (2, 0, 1))
        
        return image_array

    def __getitem__(self, i):
        try:
            image, res, raw = self.read_image(self.images[i][:-1])
        except:
            print(f"Error reading image {self.images[i]}")
            return None
        sample = {
            "image_path": self.images[i],
            "image": image,
            "is_real": np.array([1 if self.images[i][-1] == "R" else 0]),
            "original_res": res,
            "raw": raw,
        }
        return sample