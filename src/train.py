import argparse
import yaml
import pprint
import time
import os

import torch
import torch.utils.data
import lightning as L

import wandb
from pytorch_lightning.loggers import WandbLogger

from models import Backbone
from crc_datasets import CADPATH_CRC_Tiles_Dataset
from utils import create_directory


parser = argparse.ArgumentParser(
    description='Fully supervised training - MIL'
)

parser.add_argument('--config',
                    required=True,
                    help='Config file (YAML)')


args = parser.parse_args()

with open(args.config, "r") as yaml_file:
    parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)
print("Current configuration:")
pprint.pprint(parameters)
print("-"*80)

wandb_dir = create_directory(
    base_name=parameters["Registry"]["name"],
    root_dir=parameters["Registry"]["wandb_root_dir"],
)

wandb_logger = WandbLogger(
    project=parameters["Registry"]["project"],
    save_dir=os.path.join(str(parameters["Registry"]["wandb_root_dir"]), os.path.basename(wandb_dir)),
    config=parameters,
)

print("Loading train data...")
start = time.perf_counter()
train_data = CADPATH_CRC_Tiles_Dataset(
    mount_point=parameters['Data']['mount_point'],
    split="tiles-annot-train",
    is_bag=False,
)
print(f"Successfully loaded train data! Elapsed: {time.perf_counter() - start:.1f} seconds")

print("Loading validation data...")
start = time.perf_counter()
val_data = CADPATH_CRC_Tiles_Dataset(
    mount_point=parameters['Data']['mount_point'],
    split="tiles-annot-val",
    is_bag=False,
)
print(f"Successfully loaded validation data! Elapsed: {time.perf_counter() - start:.1f} seconds")

train_dataloader = torch.utils.data.DataLoader(
    batch_size=parameters['Training']['batch_size'],
    dataset=train_data,
    shuffle=True,
    num_workers=4,
)

val_dataloader = torch.utils.data.DataLoader(
    batch_size=parameters['Training']['batch_size'],
    dataset=val_data,
    num_workers=4,
)

callback_model_checkpoint = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=parameters['Registry']['checkpoints_dir'],
    filename='{epoch}',
    monitor='val/loss',
    save_last=True,
    save_top_k=1,
)

callback_early_stopping = L.pytorch.callbacks.EarlyStopping(
    monitor='val/loss',
    patience=parameters['Training']['early_stopping_patience'],
)

callback_backbone_finetuning = L.pytorch.callbacks.BackboneFinetuning(
    unfreeze_backbone_at_epoch=parameters['Training']['epochs_before_unfreeze'],
    lambda_func=lambda lr: parameters['Training']['gain_after_unfreeze'],
    backbone_initial_ratio_lr=parameters['Training']['gain_before_unfreeze'],
)

callback_learning_rate_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')

callbacks = [
    callback_model_checkpoint,
    callback_early_stopping,
    callback_backbone_finetuning,
    callback_learning_rate_monitor,
]

model = Backbone(n_classes=3, user_parameters=parameters)
print("Created new instance of model...")

torch.set_float32_matmul_precision('medium')
print("Warning! Set float32 matmul precision to 'medium'...")


trainer = L.Trainer(
    limit_train_batches=parameters['Training']['limit_train_batches'],
    max_epochs=parameters['Training']['max_epochs'],
    logger=wandb_logger,
    callbacks=callbacks,
    log_every_n_steps=parameters['Training']['log_every_n_steps'],
)

trainer.fit(model, train_dataloader, val_dataloader)
wandb.finish()

