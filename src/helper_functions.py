import time
import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from settings import Epochs, batch_size, LR, device


def train(model, train_loader, validation_loader):
    model = model.to(device)
    # defining cross entropy loss
    criterion = nn.CrossEntropyLoss()
    # creating an optimizer object performing SGD algorithm
    optimizer = optim.SGD(model.parameters(), lr=LR)
    train_loss_list_per_epoch = []
    train_loss_list_per_itr = []
    val_loss_list = []
    val_accuracy_per_epoch = []
    time_s = time.time()

    for epoch in range(Epochs):
        itr = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
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
    time_e = time.time()
    print("Training time in Mins : ", (time_e - time_s) / 60)
    # plotting the loss curve over all iteration
    print('Train loss values per iteration')
    print(train_loss_list_per_epoch)
    plt.plot(np.arange(len(train_loss_list_per_epoch)), train_loss_list_per_epoch, color='blue', label='Train')
    plt.plot(np.arange(len(val_loss_list)), val_loss_list, color='red', label='Validation')
    plt.legend()
    plt.title('Train and Validation Loss')
    plt.savefig('train_val_loss.png')
    plt.cla()
    plt.plot(np.arange(len(val_accuracy_per_epoch)), val_accuracy_per_epoch, color='green', label='Validation')
    plt.title('Validation Accuracy')
    plt.savefig('validation_accuracy.png')
    return model


def evaluation(model, validation_loader, criterion):
    model = model.to(device)
    val_loss = []
    real_label = None
    pred_label = None
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
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
