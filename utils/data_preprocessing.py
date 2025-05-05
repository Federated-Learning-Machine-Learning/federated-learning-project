import torch
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


class CIFAR100Pipeline:
    """
    CIFAR-100 pipeline for DINO ViT-S/16 training and evaluation.
    Includes advanced training-time augmentation and proper resizing.
    """

    def __init__(self, data_dir: str = "./data", val_split: float = 0.1, use_augment: bool = True):
        self.data_dir = data_dir
        self.val_split = val_split
        self.use_augment = use_augment

        # DINO ViT-S/16 uses ImageNet normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = self._build_train_transform()
        self.test_transform = self._build_test_transform()

    def _build_train_transform(self):
        """
        Build DINO-style training transform with augmentation.
        """
        transform_list = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Matches DINO ViT training
            transforms.RandomHorizontalFlip(),
        ]

        if self.use_augment:
            transform_list.insert(1, AutoAugment(policy=AutoAugmentPolicy.CIFAR10))  # Apply early

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]

        return transforms.Compose(transform_list)

    def _build_test_transform(self):
        """
        Standardized test-time transform with no augmentation.
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def run_pipeline(self):
        """
        Load, split, and transform the CIFAR-100 dataset.

        Returns:
            Tuple[Dataset, Dataset, Dataset]: trainset, valset, testset
        """
        full_trainset = CIFAR100(
            root=self.data_dir,
            train=True,
            transform=self.train_transform,
            download=True
        )

        val_size = int(len(full_trainset) * self.val_split)
        train_size = len(full_trainset) - val_size
        trainset, valset = random_split(full_trainset, [train_size, val_size])

        # Use test-like transform for validation
        valset.dataset.transform = self.test_transform

        testset = CIFAR100(
            root=self.data_dir,
            train=False,
            transform=self.test_transform,
            download=True
        )

        return trainset, valset, testset
