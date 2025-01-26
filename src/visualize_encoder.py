import argparse
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from crc_datasets import CADPATH_CRC_Tiles_Dataset
from models import Backbone  # Import the Backbone class (ensure import path is correct)


def visualize_tsne(dataset, num_samples, model_path, output_path="tsne_visualization.png"):
    """
    Loads a pre-trained model, extracts features using the backbone, and visualizes t-SNE embeddings.

    Parameters:
        dataset (torch.utils.data.Dataset): Torch dataset containing data and labels.
        num_samples (float): Proportion of samples to be drawn from the dataset (0.0 < num_samples <= 1.0).
        model_path (str): Path to the saved PyTorch model.
        output_path (str): Path to save the t-SNE visualization (.png file).

    Returns:
        None. Saves the t-SNE visualization as a .png file.
    """
    # Safety check for the number of samples parameter
    if not (0.0 < num_samples <= 1.0):
        raise ValueError("`num_samples` must be a float between 0 and 1 (exclusive).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model
    # model = torch.load(model_path, map_location=device)
    model = Backbone.load_from_checkpoint(model_path)  # Loading directly from .ckpt
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Randomly sample dataset
    total_size = len(dataset)
    subset_size = int(num_samples * total_size)
    indices = torch.randperm(total_size).tolist()[:subset_size]
    sampled_dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(sampled_dataset, batch_size=32, shuffle=False)

    # Extract features and labels
    pbar = tqdm(total=subset_size, desc="Extracting features")
    features = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            x, y = batch  # assuming dataset returns (data, label)
            x = x.to(device)
            backbone_features = model.backbone(x)  # Get 512-dimensional features from the backbone
            features.append(backbone_features.cpu().numpy())
            labels.append(y.numpy())
            pbar.update(x.size(0))
    pbar.close()

    # Stack features and labels
    features = np.vstack(features)
    labels = np.concatenate(labels)

    # Perform t-SNE on features
    print("Training TSNE... This may take a while...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features)

    # Plot t-SNE visualization, grouped by class
    print("Plotting the results!")
    plt.figure(figsize=(10, 8))
    for class_index in np.unique(labels):
        class_mask = labels == class_index
        plt.scatter(
            tsne_features[class_mask, 0],
            tsne_features[class_mask, 1],
            label=f"Class {class_index}",
            alpha=0.6
        )
    plt.title("t-SNE Visualization of Backbone Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Example usage:
# visualize_tsne(torch_dataset, 0.1, "/path/to/saved_model.pth", "output_plot.png")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="t-SNE visualization of model embeddings.")

    # Define required arguments
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset mount point"
    )
    parser.add_argument(
        "--num_samples",
        type=float,
        required=True,
        help="Fraction of dataset to sample (must be a float between 0.0 and 1.0)."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the saved PyTorch model."
    )
    parser.add_argument(
        "--output_path",
        default="tsne_visualization.png",
        help="Path to save the t-SNE scatter plot (default: tsne_visualization.png)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Load dataset

    dataset = CADPATH_CRC_Tiles_Dataset(
        mount_point=args.dataset,
        split="tiles-annot-train",
        is_bag=False,
    )

    # Call the t-SNE visualization function
    visualize_tsne(
        dataset=dataset,
        num_samples=args.num_samples,
        model_path=args.model_path,
        output_path=args.output_path
    )