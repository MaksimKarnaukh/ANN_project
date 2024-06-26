import os
import shutil
import random
import torchvision
from torchvision import datasets
import torch
import cv2
import numpy as np
import copy


def split_dataset(input_folder: str, train_folder: str, test_folder: str, validation_split: float = 0.2, random_seed: int = 42) -> None:
    """
    Split the dataset into train and test folders.
    :param input_folder: Path to the input folder
    :param train_folder: Path to the train folder
    :param test_folder: Path to the test folder
    :param validation_split: Fraction of the images to move to the test folder
    :param random_seed: Random seed (for shuffling the images)
    :return:
    """
    if random_seed is not None:
        random.seed(random_seed)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        if os.path.isdir(class_path):
            # Create train and test folders for the current class
            train_class_path = os.path.join(train_folder, class_folder)
            test_class_path = os.path.join(test_folder, class_folder)
            if not os.path.exists(train_class_path):
                os.makedirs(train_class_path)
            if not os.path.exists(test_class_path):
                os.makedirs(test_class_path)

            images = os.listdir(class_path)
            random.shuffle(images)  # Shuffle the images randomly
            num_validation = int(len(images) * validation_split)

            # Move images to train folder
            for image in images[num_validation:]:
                src = os.path.join(class_path, image)
                dst = os.path.join(train_class_path, image)
                shutil.copy(src, dst)

            # Move images to test folder
            for image in images[:num_validation]:
                src = os.path.join(class_path, image)
                dst = os.path.join(test_class_path, image)
                shutil.copy(src, dst)


def load_data(batch_size: int = 32) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the dataset.
    :param batch_size: Batch size
    :return: train_loader, validation_loader
    """
    # Access the transformation function applied during pre-training
    transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    trainset = datasets.ImageFolder(os.path.join('../15SceneData/', 'train'), transform=transform)
    print('Number of train examples:', len(trainset))

    validationset = datasets.ImageFolder(os.path.join('../15SceneData/', 'test'), transform=transform)
    print('Number of evaluation examples:', len(validationset))

    # Creating a loader object to read and load a batch of data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'Done loading data (batch size: {batch_size}) ...')

    return train_loader, validation_loader


def apply_gaussian_blur(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply Gaussian blur to the input image.
    See https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

    :param image: Input image
    :param kernel_size: Size of the Gaussian kernel
    :return: Blurred image
    """
    # Convert image tensor to numpy array
    image_np = image.permute(1, 2, 0).numpy()
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    # Convert back to tensor
    return torch.tensor(blurred_image).permute(2, 0, 1)


def load_data_blurred(batch_size: int = 32) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the dataset with each image replaced by its five different blurred versions, each with the correct label.
    :param batch_size: Batch size
    :return: train_loader, validation_loader
    """
    # Access the transformation function applied during pre-training
    transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    # Load train set
    trainset = datasets.ImageFolder(os.path.join('../15SceneData/', 'train'), transform=transform)
    train_data = []
    for image, label in trainset:
        for kernel_size in [5, 9, 13, 17, 21]:
            img = copy.deepcopy(image)
            blurred_image = apply_gaussian_blur(img, kernel_size)
            train_data.append((blurred_image, {5: 0, 9: 1, 13: 2, 17: 3, 21: 4}[kernel_size]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    print('Number of train examples:', len(train_loader.dataset))

    # Load validation set
    validationset = datasets.ImageFolder(os.path.join('../15SceneData/', 'test'), transform=transform)
    validation_data = []
    for image, label in validationset:
        for kernel_size in [5, 9, 13, 17, 21]:
            img = copy.deepcopy(image)
            blurred_image = apply_gaussian_blur(img, kernel_size)
            validation_data.append((blurred_image, {5: 0, 9: 1, 13: 2, 17: 3, 21: 4}[kernel_size]))
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Number of evaluation examples:', len(validation_loader.dataset))
    print(f'Done loading data (batch size: {batch_size}) ...')

    return train_loader, validation_loader


def apply_black_white_perturbation(image: torch.Tensor, perturbation_type: str) -> torch.Tensor:
    """
    Apply black or white perturbation to the input image.

    :param image: Input image tensor (shape: C x H x W)
    :param perturbation_type: Type of perturbation ('black' or 'white')
    :return: Perturbed image
    """
    image_np = image.permute(1, 2, 0).numpy()

    # Define a random square region with shape 10x10
    region = np.random.randint(0, min(image_np.shape[0], image_np.shape[1]) - 10, size=2)
    x, y = region

    # Apply perturbation based on the type
    if perturbation_type == 'black':
        image_np[x:x + 10, y:y + 10, :] = 0  # Set the region to black
    elif perturbation_type == 'white':
        image_np[x:x + 10, y:y + 10, :] = 255  # Set the region to white

    # Convert back to tensor and return
    return torch.tensor(image_np).permute(2, 0, 1)


def load_data_perturbation(random_seed: int = 42, batch_size: int = 32) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the dataset with each image replaced by its perturbed versions, each with the correct label.
    :param random_seed: Random seed
    :param batch_size: Batch size
    :return: train_loader, validation_loader
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Access the transformation function applied during pre-training
    transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    # Load train set
    trainset = datasets.ImageFolder(os.path.join('../15SceneData/', 'train'), transform=transform)
    train_data = []
    for image, label in trainset:
        for perturbation_type in ['black', 'white']:
            img = copy.deepcopy(image)
            perturbed_image = apply_black_white_perturbation(img, perturbation_type)
            train_data.append((perturbed_image, {'black': 0, 'white': 1}[perturbation_type]))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    print('Number of train examples:', len(train_loader.dataset))

    # Load validation set
    validationset = datasets.ImageFolder(os.path.join('../15SceneData/', 'test'), transform=transform)
    validation_data = []
    for image, label in validationset:
        for perturbation_type in ['black', 'white']:
            img = copy.deepcopy(image)
            perturbed_image = apply_black_white_perturbation(img, perturbation_type)
            validation_data.append((perturbed_image, {'black': 0, 'white': 1}[perturbation_type]))

    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print('Number of evaluation examples:', len(validation_loader.dataset))
    print(f'Done loading data (batch size: {batch_size}) ...')

    return train_loader, validation_loader


if __name__ == "__main__":
    # Split the dataset into train and test folders. Should only be executed in case the dataset is not already split.
    input_folder = "../15SceneData"
    train_folder = "train"
    test_folder = "test"
    # commented because need individual user folder permission, just copy the folders into the input folder
    # train_folder = os.path.join(input_folder, "train")
    # test_folder = os.path.join(input_folder, "test")

    split_dataset(input_folder, train_folder, test_folder, validation_split=0.2)
