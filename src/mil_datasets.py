import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import openslide
import h5py


# Assumes structure
# root
# --- slides -> file naming should follow slide_id.ext
# --- labels.csv  (slide_id, label) -> label should be a descriptive string!
# --- patches


class PatchImageDataset(Dataset):
    def __init__(
            self,
            root_dir,
            patch_size=256,
            level=0,
            transforms=None,
    ):
        self.root_dir = root_dir
        self.slides_dir = os.path.join(root_dir, "slides")
        self.patches_dir = os.path.join(root_dir, "patches")
        assert os.path.exists(os.path.join(root_dir, "labels.csv"))

        self.patch_size = patch_size
        self.level = level

        self.transforms = self.setup_transforms(transforms)

        self.labels = pd.read_csv(
            os.path.join(root_dir, "labels.csv")
        )

        self.slide_info = []
        for root, dirs, files in os.walk(self.slides_dir, topdown=False):
            for name in files:
                slide_id, extension = os.path.splitext(name)

                try:
                    self.check_files(
                        slide_path=os.path.join(root, name),
                        patch_path=os.path.join(self.patches_dir, f"{slide_id}.h5")
                    )
                except FileNotFoundError:
                    continue

                self.slide_info.append(
                    {
                        "slide_id": slide_id,
                        "slide_path": os.path.join(root, name),
                        "patch_path": os.path.join(self.patches_dir, f"{slide_id}.h5"),
                        "label": self.labels.loc[self.labels["slide_id"] == slide_id, "label"].values[0],
                    }
                )

    def __len__(self):
        return len(self.slide_info)

    def __getitem__(self, idx):
        slide_info = self.slide_info[idx]
        slide_id = slide_info["slide_id"]
        slide_path = slide_info["slide_path"]
        patch_path = slide_info["patch_path"]
        label = slide_info["label"]

        with h5py.File(patch_path, "r") as f:
            patch_coords = f["coords"][:]

        patches = []
        for coord in patch_coords:
            patch = self.process_slide(slide_path, coord, self.level, self.patch_size)
            patch = self.transforms(patch)
            patches.append(patch)
        patches = torch.stack(patches)

        patch_metadata = {
            "patch_coords": patch_coords,
            "patch_size": self.patch_size,
            "level": self.level,
        }
        output = {
            "slide_id": slide_id,
            "patches": patches,
            "label": label,
            "patch_metadata": patch_metadata,
        }
        return output

    @staticmethod
    def process_slide(slide_path, coord, level, patch_size):
        x, y = coord
        with openslide.OpenSlide(slide_path) as slide:
            patch = slide.read_region(
                location=(x, y),
                level=level,
                size=(patch_size, patch_size)
            ).convert("RGB")
        return patch

    @staticmethod
    def check_files(slide_path, patch_path):
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

    @staticmethod
    def setup_transforms(transforms):
        # TODO: data augmentation logic
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])



if __name__ == "__main__":
    dataset = PatchImageDataset(
        root_dir="/home/felipe/Projects/multiple-instance-learning/debug/dummy_data",
        patch_size=256,
        level=0
    )
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    print(batch["slide_id"])
    print(batch["patches"].shape)
    print(batch["label"])
