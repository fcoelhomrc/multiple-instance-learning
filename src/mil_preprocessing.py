import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import logging

from mil_datasets import PatchImageDataset

#TODO: patch metadata is available from PatchImageDataset, and can be used to pair the embedding to its original patch

def precompute_embeddings(
        root_dir,
        patch_size,
        level,
        encoder,
        encoder_dir,
        batch_size=32,
        device=None,
        auto_skip=False,
        logger=None,
):

    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

    output_path = os.path.join(root_dir, "embeddings", encoder_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Preparing dataset... Root: {root_dir}, Patch Size: {patch_size}, Level: {level}")
    dataset = PatchImageDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        level=level,
    )

    encoder = encoder.to(device)

    logger.info(f"Computing embeddings...")
    count = 0
    for i in range(len(dataset)):  # iterate over slides
        slide_id = dataset.get_slide_id(i)
        count += 1
        if auto_skip and os.path.exists(os.path.join(output_path, f"{slide_id}.pt")):
            logger.info(f"Embeddings for {slide_id} are already present in {output_path},"
                        f" skipping... ({count}/{len(dataset)}) ...")
            continue
        else:
            logger.info(f"Processing slide {slide_id} ({count}/{len(dataset)}) ...")

        patch_generator = dataset.get_patch_generator(i)
        buffer = []
        embeddings = []
        for patch in patch_generator:  # iterate over patches
            buffer.append(patch["patch"])

            if len(buffer) == batch_size:
                batch = torch.stack(buffer)
                batch = batch.to(device)
                with torch.no_grad():
                    batch_embedding = encoder(batch)
                embeddings.append(batch_embedding)
                buffer.clear()

        if len(buffer) > 0:
            batch = torch.stack(buffer)
            batch = batch.to(device)
            with torch.no_grad():
                batch_embedding = encoder(batch)
            embeddings.append(batch_embedding)
            buffer.clear()

        embeddings = torch.cat(embeddings, dim=0)
        logger.info(f"Saving embeddings at {os.path.join(output_path, f'{slide_id}.pt')}...")
        torch.save(embeddings, os.path.join(output_path, f"{slide_id}.pt"))
        del embeddings  # free some space before next iter





