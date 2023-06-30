import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import lightning as L

import get_data
import models
import lightning_modules

# get the data
train_data, trainloader, test_data, testloader = get_data.get_CIFAR10()

# get models
resnet_model = models.ResNet18()
clf_model = models.Classification(resnet_model)

# lightning module
autoencoder = lightning_modules.CIFAR1Module(clf_model)

# train
trainer = L.pytorch.Trainer(accelerator='auto', max_epochs=4)
trainer.fit(model=autoencoder, train_dataloaders=trainloader)

# test
test_result = trainer.test(autoencoder, dataloaders=testloader, verbose=False)
print(test_result[0])
