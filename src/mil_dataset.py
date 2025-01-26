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



class SlidePatchDataset(Dataset):
    """
    A PyTorch Dataset that extracts patches from .h5 files and assigns labels based on the slides.

    Parameters:
        root_dir (str): Root directory of the dataset.
        transform (callable, optional): Optional transform to be applied on a patch.
    """

    def __init__(self, root_dir, number_of_slides=None, transform=None):
        # Paths to required directories
        self.patches_dir = os.path.join(root_dir, "clam-outputs/patches")
        self.slides_dir = os.path.join(root_dir, "slides")
        self.labels_path = os.path.join(root_dir, "labels.csv")
        self.number_of_slides = number_of_slides

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
        self.transform = transform

        for h5_file_idx, h5_file in enumerate(self.h5_files):
            slide_id = os.path.splitext(os.path.basename(h5_file))[0]

            slide_id_number = int(slide_id.split("_")[1])
            if self.number_of_slides is not None:
                if h5_file_idx > self.number_of_slides:
                    break

            label = self.slide_labels.get(slide_id)
            if label is None:
                raise ValueError(f"Slide ID {slide_id} not found in labels CSV.")

            with h5py.File(h5_file, "r") as f:
                coords = f["coords"]
                # Iterate through each patch
                for i in range(coords.shape[0]):
                    self.patches.append({
                        "h5_file": h5_file,
                        "patch_index": i,
                        "slide_id": slide_id,
                        "label": label
                    })

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Extract patch metadata
        patch_info = self.patches[idx]
        h5_file = patch_info["h5_file"]
        patch_index = patch_info["patch_index"]
        slide_id = patch_info["slide_id"]
        label = patch_info["label"]

        # Open the .h5 file and extract the patch coordinates
        with h5py.File(h5_file, "r") as f:
            coords = f["coords"]  # (N, 2) array of coordinates
            coord = coords[patch_index]  # Get the top-left coordinate (x, y) for the patch

        # The slide image is stored in the 'slides' directory as a `.svs` file
        slide_path = os.path.join(self.slides_dir, f"{slide_id}.svs")

        # Open the slide using OpenSlide and read the region corresponding to the patch
        slide = openslide.OpenSlide(slide_path)
        x, y = coord  # Extract top-left corner from the h5 file's coords
        patch_data = slide.read_region((int(x), int(y)), 0, (256, 256))  # Level 0, patch size (256x256)

        # Convert the patch to RGB mode (if necessary)
        patch = patch_data.convert("RGB")

        # Apply optional transformations to the patch
        if self.transform:
            patch = self.transform(patch)

        # Close the slide to release resources
        slide.close()

        # Return the patch, slide ID, and label
        return patch, slide_id, label


def process_patch_ranking(model_checkpoint_path, dataset_root, output_dir, number_of_slides):
    """
    Processes patches by ranking them based on a model's predictions and writes the results
    to CSV files.

    Parameters:
        model_checkpoint_path (str): Path to the Backbone model checkpoint.
        dataset_root (str): Root path to instantiate the SlidePatchDataset.
        output_dir (str): Directory where CSV files will be written.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model from the checkpoint
    from models import Backbone  # Backbone class (PyTorch Lightning Module)
    model = Backbone.load_from_checkpoint(model_checkpoint_path)
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create SlidePatchDataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = SlidePatchDataset(root_dir=dataset_root, number_of_slides=number_of_slides, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Define class weights (for computing the rank)
    class_weights = [0, 1, 2]

    # Create buffers to store results
    global_results = []  # To store all patches with their ranks
    slide_top5 = {}  # To store top-5 ranked patches for each slide

    # Create progress bar
    pbar = tqdm(total=len(dataset), desc="Processing patches", file=sys.stdout, leave=False)

    # Iterate over the dataset
    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch_idx, (patches, slide_ids, labels) in enumerate(dataloader):
            patches = patches.to(device)

            # Compute probabilities using the model
            probabilities = torch.nn.functional.softmax(model(patches), dim=-1)

            # Compute rank for each patch
            ranks = torch.sum(probabilities * torch.tensor(class_weights, device=device), dim=-1).cpu().numpy()

            # Process each patch in the batch
            for idx, (rank, slide_id, label) in enumerate(zip(ranks, slide_ids, labels)):
                global_results.append({
                    "patch_index": batch_idx * len(patches) + idx,
                    "slide": slide_id,
                    "label": int(label),
                    "rank": float(rank)
                })

                # Maintain top-5 ranked patches for each slide
                if slide_id not in slide_top5:
                    slide_top5[slide_id] = []

                slide_top5[slide_id].append({
                    "patch_index": batch_idx * len(patches) + idx,
                    "slide": slide_id,
                    "label": int(label),
                    "rank": float(rank)
                })

            # Update the progress bar with current slide and average rank
            pbar.set_postfix(slide_id=slide_ids[0], avg_rank=ranks.mean().item())
            pbar.refresh()
            # Process the batch and update results...
            pbar.update(len(patches))  # Update progress bar

    # Sort patches in each slide by rank and keep top 5 per slide
    slide_top5_sorted = {}
    for slide_id, patch_list in slide_top5.items():
        slide_top5_sorted[slide_id] = sorted(patch_list, key=lambda x: x["rank"], reverse=True)[:5]

    # Write the full ranking results to a CSV file
    full_csv_path = os.path.join(output_dir, "patch_ranking.csv")
    with open(full_csv_path, mode="w", newline="") as full_csv_file:
        writer = csv.DictWriter(full_csv_file, fieldnames=["patch_index", "slide", "label", "rank"])
        writer.writeheader()
        writer.writerows(global_results)

    # Write the top-5 rankings to a CSV file
    top5_csv_path = os.path.join(output_dir, "top5_patch_ranking.csv")
    with open(top5_csv_path, mode="w", newline="") as top5_csv_file:
        writer = csv.DictWriter(top5_csv_file, fieldnames=["patch_index", "slide", "label", "rank"])
        writer.writeheader()
        for slide_id, top5_patches in slide_top5_sorted.items():
            writer.writerows(top5_patches)

    print(f"Processed patch rankings saved at: {output_dir}")



if __name__ == "__main__":
    import argparse

    # Define argument parser
    parser = argparse.ArgumentParser(description="Process patches by ranking them using a trained model.")

    # Define required arguments
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        required=True,
        help="Path to the Backbone model checkpoint."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root path to instantiate the SlidePatchDataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where CSV files will be written."
    )
    parser.add_argument(
        "--number_of_slides",
        type=int,
        required=True,
        help="Number of slides to be included (sequential)."
    )


    # Parse arguments
    args = parser.parse_args()

    # Call process_patch_ranking with parsed arguments
    process_patch_ranking(
        model_checkpoint_path=args.model_checkpoint_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        number_of_slides=args.number_of_slides,
    )