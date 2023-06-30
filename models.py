import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Classification(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.layer1 = nn.Linear(1000, 1000)
        self.layer2 = nn.Linear(1000, 10)

    def probabilities(self, x):
        output =  self.pretrained_model(x)
        output = F.relu(self.layer1(output))
        output = F.softmax(self.layer2(output), dim=1)
        
        return output

    def forward(self, x):
        output = self.probabilities(x)
        output = torch.argmax(output, dim=1)

        return output

def ResNet18():
    return torchvision.models.resnet18(weights='DEFAULT')