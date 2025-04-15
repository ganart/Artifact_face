import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class ArtifactDataset(Dataset):
    #Custom dataset for image
    def __init__(self, img_dir, transform=None):
        """
        Dataset initialization

        :param img_dir: path to folder with images
        :param transform: appy transform to image, like augmentation
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(self.img_dir) if f.endswith('.png')]

    def __len__(self):
        """
        :return: images count
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve an image, label, and path by index.

        :param idx: index of element
        :return: (image, label, image_path)
                    - image: Tensor.
                   - label: 0 (artifact) or 1 (no artifact).
                   - image_path: String with the full file path.
        """
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        label = int(self.images[idx].split('_')[-1].split('.')[0])
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


def get_transforms(train=True):
    """
    Create transformations for images.

    :param train: If True, apply augmentations for training; otherwise, apply only resizing for validation/testing.
    :return: v2.Compose Composition of torchvision transformations
    """
    if train:
        return v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), interpolation=v2.InterpolationMode.BILINEAR,
                                 antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-15, 15), fill=0),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            v2.RandomGrayscale(p=0.2),
            v2.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToDtype(torch.float32, scale=True),

        ])

    return v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(224, 224), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

