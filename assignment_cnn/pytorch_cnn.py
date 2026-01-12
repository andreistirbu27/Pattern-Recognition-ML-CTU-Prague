import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=28 * 28,
                            out_features=10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.fc = nn.Linear(in_features=28 * 28 * 10 // (2 * 2),
                            out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Dropout2d(p=0.10),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Dropout2d(p=0.15),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.30),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class FinetuneNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """
    def __init__(self, load_pretrained: bool = False, n_classes: int = 10):
        super().__init__()

        weights = torchvision.models.ResNet18_Weights.DEFAULT if load_pretrained else None
        self.model = torchvision.models.resnet18(weights=weights)

        # replace classifier head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)

        # freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # unfreeze last block + classifier
        for p in self.model.layer4.parameters():
            p.requires_grad = True
        for p in self.model.fc.parameters():
            p.requires_grad = True

        # imagenet normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x):
        x = (x - self._mean) / self._std
        return self.model(x)  # logits

    def train(self, mode: bool = True):
        # Important: keep BatchNorm in eval mode to avoid ruining running stats on small dataset
        super().train(mode)
        if mode:
            self.model.eval()  # freezes BN/dropout behavior in the backbone
            self.model.layer4.train()  # allow last block to train
            self.model.fc.train()  # allow head to train
        return self


def classify(model, x):
    """
    :param model:    network model object
    :param x:        (batch_size, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_size, ) torch tensor with class labels
    """
    out = model(x)  # (batch_size, num_classes)
    labels = torch.argmax(out, dim=1)
    return labels


if __name__ == '__main__':
    pass
