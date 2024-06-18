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

    Args:
        num_classes: The number of output classes in the final layer.
    Attributes:
        model: The pre-trained EfficientNet-B0 model.

    Methods:
        replace_classifier: Replace the classifier of the model with a new classifier with num_classes output units.
        set_trainable: Set the feature extraction layers of the model to be trainable or frozen. By default, the classifier part stays trainable.
    """
    def __init__(self, linear_layer_in_features: list[int], num_classes=15):
        """
        Initialize the EfficientNetModel with the EfficientNet_B0_Weights.IMAGENET1K_V1 weights.
        :param num_classes: int, number of output units in the final layer
        """
        super(EfficientNetModel, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # need to change the number of classes in the final layer
        self.replace_classifier(num_classes, linear_layer_in_features)

        # make all layers in feature extraction and classifier parts of the model trainable
        for param in self.model.parameters():
            param.requires_grad = True

    def replace_classifier(self, num_classes: int, linear_layer_in_features: list[int]) -> None:
        """
        Replace the classifier of the model with a new classifier with num_classes output units.
        :param num_classes: int, number of output units in the new classifier
        :param linear_layer_in_features: list of int, number of input units in each linear layer
        :return: None
        """

        if len(linear_layer_in_features) == 0:
            raise ValueError("linear_layer_in_features should have at least one element.")
        elif linear_layer_in_features[0] != 1280:
            raise ValueError("The first element of linear_layer_in_features should be 1280.")

        layers = [nn.Dropout(p=0.2, inplace=True)]
        in_features = linear_layer_in_features[0]

        for out_features in linear_layer_in_features[1:]:
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            in_features = out_features

        # Final layer with num_classes outputs
        layers.append(nn.Linear(in_features=in_features, out_features=num_classes))

        self.model.classifier = nn.Sequential(*layers)

    def set_trainable(self, freeze_feature_extraction: bool = False) -> None:
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
