import h5py
import os
from glob import glob
from xml.etree import ElementTree


import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from openslide import open_slide, ImageSlide

from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join("..", "samples", "PAIP", "clam-pipeline-results", "patches")

DATA_PATH = "/home/felipe/Projects/multiple-instance-learning/samples/PAIP/clam-pipeline-results/patches/001100-2019-05-00-01-01.h5"


class TileBag:

    root_dir = os.path.join("..", "samples", "PAIP")
    slides_dir = os.path.join(root_dir, "raw")
    patches_dir = os.path.join(root_dir, "clam-pipeline-results", "patches")
    annotations_dir = os.path.join(root_dir, "annotations")

    def __init__(self, id_, patch_size=256):
        self.id_ = id_
        self.wsi = open_slide(
            os.path.join(TileBag.slides_dir, f"{id_}.svs" ),
        )
        self.thumbnail = self.wsi.get_thumbnail(self.wsi.level_dimensions[-1])
        self.patch_coords = self.read_hdf5()
        patch_annotations_, resolution_ratio = self.setup_annotations()
        self.patch_annotation_resolution_ratio = resolution_ratio
        self.patch_annotations = patch_annotations_

        self.bag_size = self.patch_coords.shape[0]
        self.patch_size = patch_size

    def get_tile(self, i):
        """
        Tile-level label is assigned as:
         i) 0, if all pixel-level labels are 0 (base class)
         ii) median(x > 0), if any pixel-level labels is x > 0
        :param i: tile index
        :return: tile, label
        """
        if not (0 <= i < self.bag_size):
            return

        coords = self.patch_coords[i]
        patch_ = self.wsi.read_region(coords, level=0, size=(self.patch_size, self.patch_size))
        label_ = self.patch_annotations.read_region(
            coords, level=0,
            size=(
                int(self.patch_size * self.patch_annotation_resolution_ratio[0]),
                int(self.patch_size * self.patch_annotation_resolution_ratio[1])
            )
        )
        patch = np.array(patch_)
        label = np.array(label_)[:, :, 0]  # OpenImage returns PIL.Image (RGBA) and labels are encoded on R
        if label.max() == 0:
            return self.wrap_results(coords, patch, 0)
        return self.wrap_results(coords, patch, np.median(label[label > 0]))

    @staticmethod
    def wrap_results(coords, patch, label):
        results = {
            "coords": coords, "patch": patch, "label": label
        }
        return results

    def read_hdf5(self):
        with h5py.File(os.path.join(TileBag.patches_dir, f"{self.id_}.h5"), "r") as f:
            patch_coords = f["coords"][:]
        return patch_coords

    def read_xml(self):
        """
        <XML Tree>
        Annotations (root)
        > Annotation
          > Regions
            > Region
              > Vertices
                > Vertex
                  > X, Y
        <Label>
        nerve_without_tumor (contour): 1 yellow
        perineural_invasion_junction (line): 2 red
        nerve_without_tumor (bounding box): 11 yellow
        tumor_without_nerve (bounding box): 13 green
        nontumor_without_nerve (bounding box): 14 blue
        """
        etree = ElementTree.parse(os.path.join(TileBag.annotations_dir, f"{self.id_}.xml"))
        annotations = etree.getroot()

        full_resolution = self.wsi.level_dimensions[0]  # full resolution
        target_resolution = self.wsi.level_dimensions[-1]  # lowest resolution
        resolution_ratio = (
            target_resolution[0] / full_resolution[0],
            target_resolution[1] / full_resolution[1],
        )
        annotation_mask = np.zeros(target_resolution, dtype=np.int32)

        for annotation in annotations:
            label = int(annotation.get("Id"))
            contours = []
            regions = annotation.findall("Regions")[0]
            for region in regions.findall("Region"):
                pts = []
                vertices = region.findall("Vertices")[0]
                for vertex in vertices.findall("Vertex"):
                    x = int(vertex.get("X"))
                    y = int(vertex.get("Y"))

                    # convert from full resolution coords to target resolution coords
                    x = np.clip(int(x * resolution_ratio[0]), 0, target_resolution[0] - 1)
                    y = np.clip(int(y * resolution_ratio[1]), 0, target_resolution[1] - 1)

                    pts.append((y, x))
                contours.append(pts)

            for pts in contours:
                pts = [np.array(pts, dtype=np.int32)]
                cv2.drawContours(annotation_mask, pts, -1, label, cv2.FILLED)
        return Image.fromarray(annotation_mask.T), resolution_ratio

    def setup_annotations(self):
        _annotation_mask, resolution_ratio = self.read_xml()
        annotation_mask = ImageSlide(_annotation_mask)  # openslide wrapper
        return annotation_mask, resolution_ratio

if __name__ == "__main__":
    from tqdm import tqdm
    from skimage.transform import resize

    id_ = "001100-2019-05-00-01-01"
    tile_bag = TileBag(id_)

    res = tile_bag.get_tile(0)
    plt.imshow(res["patch"])
    plt.title(res["label"])
    plt.show()


    def stitching(tb):
        psize = tb.patch_size
        src_h, src_w = tb.wsi.level_dimensions[0]
        tgt_h, tgt_w = tb.wsi.level_dimensions[-1]
        ratio_h, ratio_w = tgt_h / src_h, tgt_w / src_w

        count = 0
        grid = np.zeros((tgt_h, tgt_w, 4))
        grid_label = np.zeros((tgt_h, tgt_w))

        for i in tqdm(range(tb.bag_size)):
            res = tb.get_tile(i)
            coords = res["coords"]
            tile = res["patch"]

            x = np.clip(int(coords[0] * ratio_h), 0, tgt_h - 1)
            y = np.clip(int(coords[1] * ratio_w), 0, tgt_w - 1)

            tx, ty = int(psize * ratio_h), int(psize * ratio_w)
            tile_rescaled = resize(tile, (tx, ty, 4),
                                   preserve_range=True)

            if res["label"] > 0:
                count += 1
            grid[x:x+tx, y:y+ty] = tile_rescaled
            grid_label[x:x+tx, y:y+ty] = res["label"]

        img = Image.fromarray(grid.astype(np.uint8).swapaxes(0, 1))
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(grid_label.swapaxes(0, 1))
        axs[0].set_title("Stitching")
        axs[1].set_title("Labels")
        plt.show()

    stitching(tile_bag)

