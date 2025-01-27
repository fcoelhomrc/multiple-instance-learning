import os, sys
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
import csv
from tqdm import tqdm
import openslide
from tqdm import tqdm


def precompute_embeddings(root_dir, model_checkpoint, output_dir, num_slides=None, patch_size=256,):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = torch.from_checkpoint(model_checkpoint).to(device)
    model.eval()

    if not hasattr(model, "preprocessor"):
        raise ValueError("Model must have a preprocessor module.")
    if not hasattr(model, "backbone"):
        raise ValueError("Model must have a backbone module.")

    mil_dataset = MILDataset(root_dir, num_slides=num_slides, patch_size=patch_size)
    dataloader = DataLoader(mil_dataset, batch_size=1, shuffle=False)

    embedding_dataset = []
    slide_id = None
    pbar = tqdm(desc="Computing embeddings", total=len(mil_dataset), file=sys.stdout, leave=False)
    for batch in dataloader:
        pbar.set_postfix(slide_id=slide_id)
        pbar.refresh()
        patches, slide_id, label = batch
        if slide_id is None:
            slide_id = slide_id
        patches = patches.to(device)
        with torch.no_grad():
            preprocessed = model.preprocessor(patches)
            embeddings = model.backbone(preprocessed)
        embeddings = embeddings.cpu().numpy()
        embedding_dataset.append(embeddings)
        pbar.update()
    embedding_dataset = np.array(embedding_dataset)
    with h5py.File(os.path.join(output_dir, f"{slide_id}.h5"), "w") as f:
        f.create_dataset("embeddings", data=embedding_dataset)


class MILDataset(Dataset):

    def __init__(self, root_dir, num_slides=None, patch_size=256):
        # Paths to required directories
        self.patches_dir = os.path.join(root_dir, "clam-outputs/patches")
        self.slides_dir = os.path.join(root_dir, "slides")
        self.labels_path = os.path.join(root_dir, "labels.csv")
        self.num_slides = num_slides
        self.patch_size = patch_size

        # Read labels CSV
        self.labels_df = pd.read_csv(self.labels_path)

        # Map slide name to label
        self.slide_labels = {row["slide_name"]: row["slide_label"]
                             for _, row in self.labels_df.iterrows()}

        # List all patch files (.h5) in the patches directory
        self.h5_files = [
            os.path.join(self.patches_dir, fname)
            for fname in os.listdir(self.patches_dir) if fname.endswith(".h5")
        ]

        # Collect all patches and their corresponding slide and labels
        self.patches = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for h5_file_idx, h5_file in enumerate(self.h5_files):
            slide_id = os.path.splitext(os.path.basename(h5_file))[0]

            if self.num_slides is not None:
                if h5_file_idx > self.num_slides:
                    break

            label = self.slide_labels.get(slide_id)
            if label is None:
                raise ValueError(f"Slide ID {slide_id} not found in labels CSV.")

            self.patches.append({
                "h5_file": h5_file,
                "slide_id": slide_id,
                "label": label
            })

    def __len__(self):
        return self.num_slides

    def __getitem__(self, idx):
        # Extract patch metadata
        patch_info = self.patches[idx]
        h5_file = patch_info["h5_file"]
        slide_id = patch_info["slide_id"]
        label = patch_info["label"]

        # Open the .h5 file and extract the patch coordinates
        with h5py.File(h5_file, "r") as f:
            coords = f["coords"]  # (N, 2) array of coordinates

        # The slide image is stored in the 'slides' directory as a `.svs` file
        slide_path = os.path.join(self.slides_dir, f"{slide_id}.svs")

        # Open the slide using OpenSlide and read the region corresponding to the patch
        slide = openslide.OpenSlide(slide_path)
        patches = []
        for coord in coords:
            x, y = coord  # Extract top-left corner from the h5 file's coords
            patch_data = slide.read_region((int(x), int(y)), 0, (self.patch_size, self.patch_size))
            # Convert the patch to RGB mode (if necessary)
            patch = patch_data.convert("RGB")
            # Apply transformations to the patch
            patch = self.transform(patch)
            patches.append(patch)
        patches = torch.stack(patches)

        # Close the slide to release resources
        slide.close()

        # Return the patch, slide ID, and label
        return patches, slide_id, label


if __name__ == "__main__":
    import argparse

    # Define argument parser
    parser = argparse.ArgumentParser(description="Precompute embeddings for slide patches.")

    # Add arguments
    parser.add_argument("root_dir", type=str, help="Root directory of the dataset.")
    parser.add_argument("model_checkpoint", type=str, help="Path to model checkpoint file.")
    parser.add_argument("output_dir", type=str, help="Directory to store the computed embeddings.")
    parser.add_argument("--num_slides", type=int, default=None,
                        help="Number of slides to process (default: None, process all slides).")
    parser.add_argument("--patch_size", type=int, default=256,
                        help="Patch size to extract from the slide (default: 256).")

    # Parse arguments
    args = parser.parse_args()

    # Call the precompute_embeddings function with parsed arguments
    precompute_embeddings(
        root_dir=args.root_dir,
        model_checkpoint=args.model_checkpoint,
        output_dir=args.output_dir,
        num_slides=args.num_slides,
        patch_size=args.patch_size
    )