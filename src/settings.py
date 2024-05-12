import torch

# batch_size
batch_size = 15
# input shape
input_shape = [224, 224]

# mean and standard deviation of each channel of input for performing normalization
# mean_std_normalization = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# mean_std_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# learning rate
LR = 1e-3
# number of epochs for training the model
Epochs = 20
# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
