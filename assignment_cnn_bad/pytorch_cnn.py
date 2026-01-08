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
        # CNN with stride=1 + maxpool, plus dropout before FC
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28->14->7
        self.drop = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.drop(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FinetuneNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """
    def __init__(self, load_pretrained=False):
        super().__init__()
        weights = None
        if load_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT

        self.model = torchvision.models.resnet18(weights=weights)

        # replace the last FC layer for 2 classes
        in_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feats, 2)

        # freeze everything except the last FC (recommended baseline for small datasets)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.fc.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def classify(self, x):
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


def classify(model, x):
    """
    :param model:    network model object
    :param x:        (batch_size, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_size, ) torch tensor with class labels
    """
    logits = model(x)  # (batch, 10) log-probs
    labels = torch.argmax(logits, 1)  # (batch,)
    return labels


if __name__ == '__main__':
    pass
