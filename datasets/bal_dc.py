import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import rasterio


def random_click(mask: np.ndarray, point_labels: int = 1):
    indices = np.argwhere(mask == point_labels)
    if indices.size == 0:
        point_labels = 1
        indices = np.argwhere(mask == point_labels)
    return point_labels, indices[np.random.randint(len(indices))]


class BalDc(Dataset):
    def __init__(self, data_path, ground_truth_csv, transform=None):
        csv_path = (
            ground_truth_csv
            if os.path.isabs(ground_truth_csv)
            else os.path.join(data_path, ground_truth_csv)
        )
        df = pd.read_csv(csv_path, encoding="gbk")
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        msk_path = os.path.join(self.data_path, self.label_list[index])

        with rasterio.open(img_path) as src:
            bands = src.read()
        img = torch.from_numpy(bands).float()
        mask = Image.open(msk_path)

        img, mask = self.transform(img, mask)
        img = img.float() / 255.0

        label_array = np.array(mask).astype(np.int64)
        label_mapped = torch.from_numpy(label_array).long()

        valid_labels = torch.unique(label_mapped)
        point_label = valid_labels[torch.randint(0, valid_labels.size(0), (1,))].item()
        point_label, pt = random_click(label_mapped.numpy(), point_label)

        name = os.path.splitext(name.split("/")[-1])[0]

        return {
            "image": img,
            "label": label_mapped,
            "p_label": point_label,
            "pt": pt,
            "image_meta_dict": {"filename_or_obj": name},
        }
