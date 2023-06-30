import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class CIFARModule(L.LightningModule):
    def __init__(self, model, lr=1e-3, momentum=0.9):
        super().__init__()
        self.model = model
        self.loss_module = nn.CrossEntropyLoss()
        self.lr = lr
        self.momentum = momentum

    def forward(self, imgs):
        return torch.argmax(self.model(imgs), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        one_hot_labels = F.one_hot(labels, 10)

        probs = self.model.probabilities(inputs)
        loss = self.loss_module(probs, one_hot_labels.float())
        preds = self.model(inputs)
        acc = (preds == labels).float().mean()

        # log accuracy and loss
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self.model(inputs)
        acc = (preds == labels).float().mean()

        self.log("test_acc", acc)