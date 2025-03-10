import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 38)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, labels=None):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn1_1(self.conv1_1(x1)))
        x1 = self.pool(x1)
        
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = F.relu(self.bn2_1(self.conv2_1(x2)))
        x2 = self.pool(x2)
        
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = F.relu(self.bn3_1(self.conv3_1(x3)))
        x3 = self.pool(x3)
        
        x = self.global_pool(x3)
        x = x.view(-1, 128)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            labels = torch.argmax(labels, dim=1)
            labels = labels.long()
            loss = loss_fn(x, labels)
            return loss, x
        return x