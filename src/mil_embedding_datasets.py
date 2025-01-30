import os
import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset
import torchvision.transforms
import openslide
import h5py
from tqdm import tqdm


# Assumes structure
# root
# --- slides -> file naming should follow slide_id.ext
# --- labels.csv  (slide_id, label) -> label should be a descriptive string!
# --- label_info.json -> mapping int: str describing the labels
# --- splits.csv  (slide_id, split) -> split should be "train", "test", or "validation"
# --- patches
# --- embeddings
# --- --- encoder_A -> precomputed embeddings with encoder A
# --- --- encoder_B -> precomputed embeddings with encoder B

class PatchEmbeddingDataset(Dataset):
    def __init__(
            self,
            root_dir,
            encoder_dir,
            split,
    ):
        self.root_dir = root_dir
        self.encoder_dir = os.path.join(root_dir, "embeddings", encoder_dir)
        assert os.path.exists(os.path.join(root_dir, "labels.csv"))
        assert os.path.exists(os.path.join(root_dir, "splits.csv"))

        self.split = split

        assert os.path.exists(os.path.join(root_dir, "label_info.json"))
        with open(os.path.join(root_dir, "label_info.json"), "r") as f:
            self.label_info = json.load(f)
        assert isinstance(self.label_info, dict)
        self.num_classes = len(self.label_info)


        self.labels = self.preprocess_labels(
            pd.read_csv(
                os.path.join(root_dir, "labels.csv")
            ),
            label_info=self.label_info,
        )

        self.splits = self.preprocess_splits(
            pd.read_csv(
                os.path.join(root_dir, "splits.csv")
            ),
            split=split
        )

        self.embedding_info = []
        for root, dirs, files in os.walk(self.encoder_dir, topdown=False):
            for name in files:
                slide_id, extension = os.path.splitext(name)
                try:
                    self.check_files(
                        embedding_path=os.path.join(root, name),
                    )
                except FileNotFoundError:
                    continue

                if extension != ".pt":
                    continue  # silently ignore foreign files

                if slide_id not in self.splits:
                    continue

                self.embedding_info.append(
                    {
                        "slide_id": slide_id,
                        "embedding_path": os.path.join(root, name),
                        "label": self.labels.get(slide_id, None),
                    }
                )

    def __len__(self):
        return len(self.embedding_info)

    def __getitem__(self, idx):
        slide_id = self.embedding_info[idx]["slide_id"]
        embedding_path = self.embedding_info[idx]["embedding_path"]
        label = self.embedding_info[idx]["label"]
        embedding = torch.load(embedding_path, weights_only=True, map_location="cpu")
        output = {
            "slide_id": slide_id,
            "embedding": embedding,
            "label": label,
        }
        return output

    @staticmethod
    def check_files(embedding_path):
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    @staticmethod
    def preprocess_labels(labels, label_info):
        assert isinstance(labels, pd.DataFrame)
        assert isinstance(label_info, dict)
        if labels["label"].dtype == object:  # labels as strings
            labels["label"] = labels["label"].map(label_info)
        if labels["label"].isnull().any():  # check for unmapped labels
            raise ValueError(f"Detected unmapped labels! "
                             f"Mapping: {list(label_info.keys())}, "
                             f"Values: {list(labels['label'].unique())}")
        labels = dict(
            zip(labels["slide_id"], labels["label"])
        )
        return labels

    @staticmethod
    def preprocess_splits(split_info, split):
        assert isinstance(split_info, pd.DataFrame)
        assert split in ["train", "test", "validation"]
        split_info["split"] = split_info["split"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )
        split_info["split"] = split_info["split"].apply(
            lambda x: "validation" if x in ["val", "eval"] else x
        )
        split_info = split_info[split_info["split"] == split].reset_index(drop=True)
        split_info = dict(
            zip(split_info["slide_id"], split_info["split"])
        )
        return split_info

    def get_slide_id(self, idx):
        return self.embedding_info[idx]["slide_id"]

    def get_num_classes(self):
        return self.num_classes

    def get_label_info(self):
        return self.label_info

    def get_split(self):
        return self.split

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = PatchEmbeddingDataset(
        root_dir="/home/felipe/Projects/multiple-instance-learning/debug/dummy_data",
        encoder_dir="resnet34_ImageNet",
        split="train"
    )

    test_dataset = PatchEmbeddingDataset(
        root_dir="/home/felipe/Projects/multiple-instance-learning/debug/dummy_data",
        encoder_dir="resnet34_ImageNet",
        split="test"
    )


    validation_dataset = PatchEmbeddingDataset(
        root_dir="/home/felipe/Projects/multiple-instance-learning/debug/dummy_data",
        encoder_dir="resnet34_ImageNet",
        split="validation"
    )

    all_datasets = [train_dataset, test_dataset, validation_dataset]
    for dataset in all_datasets:
        print(f"num_classes: {dataset.get_num_classes()}")
        print(f"split: {dataset.get_split()}")
        print(f"label_info: {dataset.get_label_info()}")

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=2)

    for batch in train_dataloader:
        print(f"slide_id: {batch['slide_id']}, "
              f"label: {batch['label']}, "
              f"embedding shape: {batch['embedding'].shape}")

    for batch in test_dataloader:
        print(f"slide_id: {batch['slide_id']}, "
              f"label: {batch['label']}, "
              f"embedding shape: {batch['embedding'].shape}")

    for batch in validation_dataloader:
        print(f"slide_id: {batch['slide_id']}, "
              f"label: {batch['label']}, "
              f"embedding shape: {batch['embedding'].shape}")


