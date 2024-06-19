import time
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from settings import device
from itertools import product
from models import EfficientNetModel
from dataset import load_data
import pandas as pd


def train(model: any, param_dict: dict, train_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
          output_path: str = '../output/', verbose: int = 1) -> any:
    """
    Function to train the model.
    :param model: Model to train
    :param param_dict: Dictionary of hyperparameters
    :param train_loader: DataLoader for training data
    :param validation_loader: DataLoader for validation data
    :param output_path: Path to save the plots
    :param verbose: whether to print extra output
    :return: Trained model
    """

    try:
        param_learning_rate = param_dict['learning_rate']
        param_batch_size = param_dict['batch_size']  # isn't used in this function, but needed for file naming
        param_epochs = param_dict['epochs']
        param_linear_layer_in_features = param_dict['linear_layer_in_features']  # isn't used in this function, but needed for file naming
        param_optimizer = param_dict['optimizer']
    except KeyError as e:
        print(f"Missing parameter: {e}")
        return

    model = model.to(device)
    # defining cross entropy loss
    criterion = nn.CrossEntropyLoss()

    # creating an optimizer object
    optimizer = None
    if param_optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=param_learning_rate)
    elif param_optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=param_learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {param_optimizer}")

    train_loss_list_per_epoch = []
    train_loss_list_per_itr = []
    val_loss_list = []
    val_accuracy_per_epoch = []
    best_epoch_accuracy = 0.0
    time_s = time.time()

    for epoch in range(param_epochs):
        model.train()
        start = time.time()
        itr = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)

            # clearing old gradients from the last step
            optimizer.zero_grad()
            # computing the derivative of the loss w.r.t. the parameters
            loss.backward()
            # optimizer takes a step in updating parameters based on the gradients of the parameters.
            optimizer.step()
            if itr % 10 == 0:
                train_loss_list_per_itr.append(loss.item())
            itr += 1
        train_loss_list_per_epoch.append(np.mean(train_loss_list_per_itr))
        # Evaluate model for each update iteration
        eval_loss, eval_acc = evaluation(model, validation_loader, criterion)

        val_loss_list.append(eval_loss)
        val_accuracy_per_epoch.append(eval_acc)
        if eval_acc > best_epoch_accuracy:
            best_epoch_accuracy = eval_acc
        end = time.time()

        if verbose:
            print(
                f'Epoch {epoch + 1}/{param_epochs} done in {(end - start):.2f} seconds ; Train Loss: {train_loss_list_per_epoch[-1]:.4f}')
            print(f'Validation Loss: {eval_loss:.4f}, Validation Accuracy: {eval_acc:.4f}')
            print('---')

    time_e = time.time()
    if verbose > 1:
        print("Training time in Mins : ", (time_e - time_s) / 60)
        print('Train loss values per epoch:')
        print(train_loss_list_per_epoch)
    # plotting the loss curve over all iteration
    plt.plot(np.arange(len(train_loss_list_per_epoch)), train_loss_list_per_epoch, color='blue', label='Train')
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, color='red', label='Validation')
    plt.legend()
    plt.title('Train and Validation Loss')
    settings_string = f"_{param_learning_rate}_{param_batch_size}_{param_epochs}_{param_linear_layer_in_features}_{param_optimizer}"
    plt.savefig(output_path + f'train_val_loss{settings_string}.png')
    plt.cla()
    plt.plot(np.arange(len(val_accuracy_per_epoch)), val_accuracy_per_epoch, color='green', label='Validation')
    plt.title('Validation Accuracy')
    plt.savefig(output_path + f'validation_accuracy{settings_string}.png')
    plt.cla()
    return model, best_epoch_accuracy, val_accuracy_per_epoch[-1]


def evaluation(model: any, validation_loader: torch.utils.data.DataLoader, criterion: any) -> tuple[float, float]:
    """
    Function to evaluate the model.
    :param model: Model to evaluate
    :param validation_loader: DataLoader for validation data
    :param criterion: Loss function
    :return: Mean loss and accuracy
    """
    model.eval()  # https://stackoverflow.com/questions/53879727/pytorch-how-to-deactivate-dropout-in-evaluation-mode#:~:text=For%20instance%2C%20while%20calling%20model,Dropout%20module.
    model = model.to(device)
    val_loss = []
    real_label = None
    pred_label = None
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        val_loss.append(loss.item())

        _, preds = torch.max(outputs, 1)
        if real_label is None:
            real_label = labels.data
            pred_label = preds
        else:
            real_label = torch.cat((real_label, labels.data), dim=0)
            pred_label = torch.cat((pred_label, preds), dim=0)
        del inputs
        del labels

    real_label = real_label.detach().cpu().numpy()
    pred_label = pred_label.detach().cpu().numpy()

    report = classification_report(real_label, pred_label)
    eval_acc = float(report.split('accuracy')[1].split(' ')[27])

    return np.mean(val_loss), eval_acc


def gridsearch(param_grid: dict, output_path: str, num_classes: int):
    """
    Function to perform grid search.
    :param param_grid: Dictionary of hyperparameters to search
    :param output_path: Path to save the plots
    :return: Best model
    """

    best_params = None
    best_model = None
    best_accuracy = 0.0

    results_list = []
    param_iter = 0

    print('Device used:', device)

    for batch_size in param_grid['batch_size']:

        # Load the data using the current batch size
        train_loader, validation_loader = load_data(batch_size=batch_size)

        for learning_rate, epochs, linear_layer_in_features, optimizer in product(
                param_grid['learning_rate'], param_grid['epochs'],
                param_grid['linear_layer_in_features'], param_grid['optimizer']):


            # Create the model with the current hyperparameters
            model = EfficientNetModel(num_classes=num_classes, linear_layer_in_features=linear_layer_in_features).model

            # Train and evaluate the model
            trained_model, best_epoch_accuracy, last_epoch_accuracy = train(model, {'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': epochs, 'linear_layer_in_features': linear_layer_in_features, 'optimizer': optimizer}, train_loader, validation_loader,
                                  output_path, verbose=0)

            # Append results to the list
            results_list.append([
                len(results_list),
                learning_rate,
                batch_size,
                epochs,
                linear_layer_in_features,
                optimizer,
                best_epoch_accuracy,
                last_epoch_accuracy
            ])

            # Update the best parameters if current accuracy is better
            if last_epoch_accuracy > best_accuracy:
                best_accuracy = last_epoch_accuracy
                best_params = {
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'linear_layer_in_features': linear_layer_in_features,
                    'optimizer': optimizer,
                    'batch_size': batch_size
                }
                best_model = trained_model

            param_iter += 1
            print(
                f"Completed Training ({param_iter}) with {{learning_rate={learning_rate}, epochs={epochs}, linear_layer_in_features={linear_layer_in_features}, optimizer={optimizer}, batch_size={batch_size}}}")

    print('Grid search completed.')
    results_df = pd.DataFrame(results_list, columns=['index', 'lr', 'batch_size', 'epochs', 'linear_layers', 'optimizer', 'best_epoch_acc', 'last_epoch_acc'])
    print("Saving grid search results to", output_path + 'grid_search_results.xlsx')
    results_df.to_excel(output_path + 'grid_search_results.xlsx', index=False)

    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_accuracy}")
    return best_model
