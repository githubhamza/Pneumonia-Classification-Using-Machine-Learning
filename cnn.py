import torch
import torch.nn as nn


class feature_extractor(nn.Module):
    def __init__(self, in_size):
        super(feature_extractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_size, 8, (3, 3), padding='same'),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), padding='same'),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        return

    def forward(self, x):
        out = self.features(x)
        return out

class binary_classifier(nn.Module):
    def __init__(self, in_size):
        super(binary_classifier, self).__init__()
        self.feature_extractor = feature_extractor(in_size)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        return

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out