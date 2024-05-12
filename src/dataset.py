import os
import shutil
import random


def split_dataset(input_folder, train_folder, test_folder, validation_split=0.2, random_seed=42):
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


if __name__ == "__main__":
    input_folder = "../15SceneData"
    train_folder = "train"
    test_folder = "test"
    # commented because need individual user folder permission, just copy the folders into the input folder
    # train_folder = os.path.join(input_folder, "train")
    # test_folder = os.path.join(input_folder, "test")

    split_dataset(input_folder, train_folder, test_folder, validation_split=0.2)
