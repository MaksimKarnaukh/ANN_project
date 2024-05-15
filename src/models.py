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
        """
        Initialize the EfficientNetModel with the EfficientNet_B0_Weights.IMAGENET1K_V1 weights.
        :param num_classes: int, number of output units in the final layer
        """
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # need to change the number of classes in the final layer
        self.model.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=num_classes))

        # make all layers in feature extraction and classifier parts of the model trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def replace_classifier(self, num_classes) -> None:
        """
        Replace the classifier of the model with a new classifier with num_classes output units.
        :param num_classes: int, number of output units in the new classifier
        :return: None
        """
        self.model.classifier = torch.nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                    nn.Linear(in_features=1280, out_features=num_classes))

    def set_trainable(self, freeze_feature_extraction=False) -> None:
        """
        Set the feature extraction layers of the model to be trainable or frozen.
        :param freeze_feature_extraction: bool, if True, freeze the feature extraction layers of the model
        :return: None
        """
        if freeze_feature_extraction:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True


if __name__ == "__main__":
    # quick test
    model = EfficientNetModel()
    print(model.model)
    x = torch.randn(1, 3, 224, 224)
    print(model.model(x).shape)
