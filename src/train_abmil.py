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

from models import ABMIL
# from crc_datasets import CADPATH_CRC_Tiles_Dataset
from mil_embedding_datasets import PatchEmbeddingDataset
from utils import create_directory


parser = argparse.ArgumentParser(
    description='Weakly supervised training - ABMIL'
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

train_dataset = PatchEmbeddingDataset(
    root_dir=parameters["Data"]["mount_point"],
    encoder_dir=parameters["Data"]["encoder"],
    split="train"
)

test_dataset = PatchEmbeddingDataset(
    root_dir=parameters["Data"]["mount_point"],
    encoder_dir=parameters["Data"]["encoder"],
    split="test"
)

validation_dataset = PatchEmbeddingDataset(
    root_dir=parameters["Data"]["mount_point"],
    encoder_dir=parameters["Data"]["encoder"],
    split="validation"
)

all_datasets = [train_dataset, test_dataset, validation_dataset]
for dataset in all_datasets:
    print(f"num_classes: {dataset.get_num_classes()}")
    print(f"split: {dataset.get_split()}")
    print(f"label_info: {dataset.get_label_info()}")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4)

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

callback_learning_rate_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='step')

callbacks = [
    callback_model_checkpoint,
    callback_early_stopping,
    callback_learning_rate_monitor,
]

model = ABMIL(n_classes=3, user_parameters=parameters)
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

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(ckpt_path="best", dataloaders=test_dataloader)

wandb.finish()

