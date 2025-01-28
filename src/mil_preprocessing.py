import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
import openslide
import h5py
from tqdm import tqdm

from mil_datasets import PatchImageDataset


def precompute_embeddings(
        root_dir,
        patch_size,
        level,
        encoder_file,
        batch_size=32,
):
    dataset = PatchImageDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        level=level,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if not os.path.exists(os.path.join(root_dir, "embeddings")):
        os.mkdir(os.path.join(root_dir, "embeddings"))

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    # encoder
    # -- a checkpoint .ckpt
    # -- a pytorch model .pth
    assert os.path.exists(encoder_file)
    # TODO: this should be generic. We can have torch, lightning, or hugging face models...
    from models import Backbone
    encoder = Backbone.load_from_checkpoint(encoder_file)
    encoder.eval()
    encoder = encoder.to(device)

    pbar = tqdm(desc="Computing embeddings...", total=len(dataset), leave=False)
    for batch in dataloader:
        slide_id = batch["slide_id"]
        patches = batch["patches"]

        num_iter = len(patches) // batch_size
        remainder = len(patches) % batch_size
        embeddings = []
        pbar.set_postfix(slide_id=slide_id, step="encoding")
        for i in range(num_iter):
            patch_batch = patches[i*batch_size:(i+1)*batch_size]
            patch_batch = patch_batch.to(device)
            with torch.no_grad():
                preprocessed = encoder.preprocessor(patch_batch)
                batch_embedding = encoder.backbone(preprocessed)
            embeddings.append(batch_embedding.cpu().numpy().squeeze())
        if remainder > 0:
            patch_batch = patches[num_iter*batch_size:]
            patch_batch = patch_batch.to(device)
            with torch.no_grad():
                preprocessed = encoder.preprocessor(patch_batch)
                batch_embedding = encoder.backbone(preprocessed)
            embeddings.append(batch_embedding.cpu().numpy().squeeze())
        del patches
        pbar.set_postfix(slide_id=slide_id, step="saving")
        embeddings = np.stack(embeddings)
        with h5py.File(os.path.join(root_dir, "embeddings", f"{slide_id}.h5"), "w") as f:
            _ = f.create_dataset("embeddings", data=embeddings)
            h5py_patch_metadata_dataset = f.create_dataset("patch_metadata",
                                                           data=batch["patch_metadata"]["patch_coords"])
            h5py_patch_metadata_dataset.attrs["patch_size"] = batch["patch_metadata"]["patch_size"]
            h5py_patch_metadata_dataset.attrs["level"] = batch["patch_metadata"]["level"]
        pbar.update()






