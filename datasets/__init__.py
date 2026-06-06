import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

from .bal_dc import BalDc
from .bal_dc_simmim import BalDcSimMIM


def get_dataloader(args):
    if args.dataset == "bal_dc":
        transform_train = SegmentationTransform_4band(args.image_size, mode="train")
        transform_test = SegmentationTransform_4band(args.image_size, mode="test")
        train_dataset = BalDc(
            args.data_path,
            args.train_ground_truth_csv,
            transform=transform_train,
        )
        test_dataset = BalDc(
            args.data_path,
            args.test_ground_truth_csv,
            transform=transform_test,
        )

        nice_train_loader = DataLoader(
            train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True
        )
        nice_test_loader = DataLoader(
            test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True
        )
    elif args.dataset == "bal_dc_simmim":
        simmim_dataset = BalDcSimMIM(
            args,
            data_path=args.data_path,
            model_patch_size=16,
        )

        nice_train_loader = DataLoader(
            simmim_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True
        )
        return nice_train_loader
    else:
        raise ValueError(
            f"Unknown args.dataset={args.dataset!r}; "
            "expected 'bal_dc' or 'bal_dc_simmim'."
        )

    return nice_train_loader, nice_test_loader


class SegmentationTransform_4band:
    def __init__(self, image_size, mode="train"):
        self.image_size = image_size
        self.mode = mode

        self.resize_img = transforms.Resize((image_size, image_size))
        self.resize_mask = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)

    def __call__(self, img, mask):
        img = self.resize_img(img)
        mask = self.resize_mask(mask)

        if self.mode == "train":
            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)

        return img, mask
