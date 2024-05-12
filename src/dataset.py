import os
import shutil
import random
import torchvision
from torchvision import datasets
import torch
from settings import batch_size, mean_std_normalization, input_shape


def split_dataset(input_folder, train_folder, test_folder, validation_split=0.2, random_seed=42) -> None:
    """
    Split the dataset into train and test folders
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


def load_data() -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the dataset
    :return: train_loader, validation_loader
    """
    # Access the transformation function applied during pre-training
    transform = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    trainset = datasets.ImageFolder(os.path.join('./data/15SceneData/', 'train'), transform=transform)
    print('Number of train examples:', len(trainset))

    validationset = datasets.ImageFolder(os.path.join('./data/15SceneData/', 'test'), transform=transform)
    print('Number of evaluation examples:', len(validationset))

    # Creating a loader object to read and load a batch of data
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, validation_loader


if __name__ == "__main__":
    input_folder = "../15SceneData"
    train_folder = "train"
    test_folder = "test"
    # commented because need individual user folder permission, just copy the folders into the input folder
    # train_folder = os.path.join(input_folder, "train")
    # test_folder = os.path.join(input_folder, "test")

    split_dataset(input_folder, train_folder, test_folder, validation_split=0.2)
