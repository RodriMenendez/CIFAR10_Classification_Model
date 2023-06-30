import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import get_data
import models
import lightning_modules

import time

# get the data
train_data, trainloader, test_data, testloader = get_data.get_CIFAR10()

# get models
resnet_model = models.ResNet18()
clf_model = models.Classification(resnet_model)

# wandb
wandb_config = {
        "dataset": "CIFAR10",
        "architecture": "Resnet18+HiddenLayer"
    }

wandb_logger = WandbLogger(project='CIFAR10-Resnet18-Classification', config=wandb_config)

# lightning module
autoencoder = lightning_modules.CIFARModule(clf_model)

# train
trainer = L.pytorch.Trainer(accelerator='auto', max_epochs=4, logger=wandb_logger)
trainer.fit(model=autoencoder, train_dataloaders=trainloader)

# test
test_result = trainer.test(autoencoder, dataloaders=testloader, verbose=False)

# save model
timestr = time.strftime("%Y%m%d-%H%M%S")

path = './clf_model' + timestr
torch.save(clf_model.state_dict(), path)