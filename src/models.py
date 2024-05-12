import torch
from torch import nn
import torchvision
from torchvision.models import EfficientNet, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b0


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=15):
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # You should fine-tune a convolutional neural network architecture, pre-trained on the
        # ImageNet dataset from the Pytorch deep learning library, on the 15 scene dataset. For
        # this assignment you will use a pre-trained EfficientNet-B0 architecture from the Pytorch deep
        # learning library. You should make all layers in feature extraction and classifier parts of the
        # model trainable. When you load the model, you should also load the same transformation function
        # applied during the pre-training phase and consider it as the transformation function
        # in your supervised and self-supervised schemes. You can access the transformation via
        # torchvision.models.EfficientNetB0Weights.IMAGENET1KV1.transforms().

        # need to change the number of classes in the final layer, beware self.model has no attribute _fc
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
