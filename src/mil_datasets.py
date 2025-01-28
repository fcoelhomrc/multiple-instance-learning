import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
import openslide
import h5py
from tqdm import tqdm


# Assumes structure
# root
# --- slides -> file naming should follow slide_id.ext
# --- labels.csv  (slide_id, label) -> label should be a descriptive string!
# --- patches


class PatchImageDataset:
    def __init__(
            self,
            root_dir,
            patch_size=512,
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
                        "num_patches": self.check_num_patches(os.path.join(self.patches_dir, f"{slide_id}.h5"))
                        # TODO: not optimal, because we need to re-open the .h5 file later
                    }
                )

    def __len__(self):
        return len(self.slide_info)

    def get_patch_generator(self, idx):
        slide_info = self.slide_info[idx]
        slide_id = slide_info["slide_id"]
        slide_path = slide_info["slide_path"]
        patch_path = slide_info["patch_path"]
        label = slide_info["label"]

        with h5py.File(patch_path, "r") as f:
            patch_coords = f["coords"][:]

        with openslide.OpenSlide(slide_path) as slide:
            for coord in patch_coords:
                patch = self.process_slide(slide, coord, self.level, self.patch_size)
                patch = self.transforms(patch)
                output = {
                    "slide_id": slide_id,
                    "patch": patch,
                    "label": label,
                }
                yield output

    @staticmethod
    def process_slide(slide, coord, level, patch_size):
        x, y = coord
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
    def check_num_patches(patch_path):
        with h5py.File(patch_path, "r") as f:
            num_patches = f["coords"].shape[0]
        return num_patches

    @staticmethod
    def setup_transforms(transforms):
        # TODO: data augmentation logic
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def get_slide_id(self, idx):
        return self.slide_info[idx]["slide_id"]

    def get_num_patches(self, idx):
        return self.slide_info[idx]["num_patches"]


if __name__ == "__main__":
    dataset = PatchImageDataset(
        root_dir="/home/felipe/Projects/multiple-instance-learning/debug/dummy_data",
        patch_size=512,
        level=0
    )


    def plot_random_samples_from_tensor(tensor, num_samples):
        import matplotlib.pyplot as plt
        import numpy as np
        assert len(tensor.shape) == 4, "Tensor must have shape (Batch, Channels, Height, Width)"
        batch_size, channels, height, width = tensor.shape
        grid_size = int(np.ceil(np.sqrt(batch_size)))  # Size of the grid (rows/cols)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < num_samples:
                # Convert tensor picture to numpy for matplotlib
                img = tensor[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis("off")
            else:
                # Hide unused subplots in grid
                ax.axis("off")
        plt.tight_layout()
        plt.show()


    for i in range(len(dataset)):  # iterate over slides
        patch_generator = dataset.get_patch_generator(i)
        patches = []
        for batch in patch_generator:
            print(batch["slide_id"])
            print(batch["patch"].shape)
            print(batch["label"])
            patches.append(batch["patch"])
            if len(patches) == 16:
                break
        plot_random_samples_from_tensor(torch.stack(patches), 16)
