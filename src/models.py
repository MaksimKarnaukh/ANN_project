import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetModel:
    """
    We fine-tune a convolutional neural network architecture, pre-trained on the
    ImageNet dataset from the Pytorch deep learning library (on the 15 scene dataset).
    We will use a pre-trained EfficientNet-B0 architecture from the Pytorch deep
    learning library. We make all layers in feature extraction and classifier parts of the
    model trainable.
    """
    def __init__(self, num_classes=15):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # need to change the number of classes in the final layer
        self.model.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=num_classes))

        # make all layers in feature extraction and classifier parts of the model trainable
        for param in self.model.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # quick test
    model = EfficientNetModel()
    print(model.model)
    x = torch.randn(1, 3, 224, 224)
    print(model.model(x).shape)
