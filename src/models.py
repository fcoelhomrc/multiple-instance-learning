
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import torchmetrics

import lightning as L


# fully-supervised fine-tuning
class Backbone(L.LightningModule):

    def __init__(self, n_classes, user_parameters):
        super().__init__()

        self.n_classes = n_classes
        self.user_parameters = user_parameters

        self.backbone_weights = torchvision.models.ResNet34_Weights.DEFAULT
        self.preprocessor = self.backbone_weights.transforms()
        self.backbone = torchvision.models.resnet34(weights=self.backbone_weights)
        self.n_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.n_features, self.n_classes)

        match self.user_parameters['Loss_Function']['loss_function']:
            case 'cross_entropy':
                self.loss_function = F.cross_entropy
            case 'qwk':
                from WeightedKappaLoss import WeightedKappaLoss
                self.loss_function = WeightedKappaLoss(self.n_classes, mode='quadratic')
            case _:
                self.loss_function = F.cross_entropy  # defaults to cross entropy

        self.save_hyperparameters()  # wandb

    def forward(self, x):
        x_processed = self.preprocessor(x)
        return self.backbone(x_processed)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=self.user_parameters['Optimizer']['lr'],
            weight_decay=self.user_parameters['Optimizer']['weight_decay'],
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        probas = F.softmax(logits, dim=1)
        y_hat = probas.argmax(dim=1)

        accuracy = torchmetrics.functional.accuracy(
            y_hat, y, task='multiclass', num_classes=self.n_classes
        )

        qwk = torchmetrics.functional.cohen_kappa(
            y_hat, y, task='multiclass', num_classes=self.n_classes,
            weights='quadratic'
        )

        recall = torchmetrics.functional.recall(
            y_hat, y, task='multiclass', num_classes=self.n_classes,
        )

        self.log("train/loss", loss)  # wandb
        self.log("train/accuracy", accuracy)  # wandb
        self.log("train/recall", recall)  # wandb
        self.log("train/qwk", qwk)  # wandb
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        probas = F.softmax(logits, dim=1)
        y_hat = probas.argmax(dim=1)

        accuracy = torchmetrics.functional.accuracy(
            y_hat, y, task='multiclass', num_classes=self.n_classes
        )

        qwk = torchmetrics.functional.cohen_kappa(
            y_hat, y, task='multiclass', num_classes=self.n_classes,
            weights='quadratic'
        )

        recall = torchmetrics.functional.recall(
            y_hat, y, task='multiclass', num_classes=self.n_classes,
        )

        self.log("val/loss", loss)  # wandb
        self.log("val/accuracy", accuracy)  # wandb
        self.log("val/recall", recall)  # wandb
        self.log("val/qwk", qwk)  # wandb


