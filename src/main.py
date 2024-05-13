from dataset import load_data
from models import EfficientNetModel
from helper_functions import train


if __name__ == "__main__":
    # Loading data
    train_loader, validation_loader = load_data()
    # creating a model
    model = EfficientNetModel()
    # training the model
    model = train(model, train_loader, validation_loader)
