import torch
from torch import nn
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetModel(nn.Module):

    def __init__(self, num_classes=15):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # need to change the number of classes in the final layer
        self.model.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=num_classes))

        # make all layers in feature extraction and classifier parts of the model trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = EfficientNetModel()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)
