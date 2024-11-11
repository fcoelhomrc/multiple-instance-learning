import os

import numpy as np
import cv2
from skimage.segmentation import clear_border

from openslide import open_slide
from PIL import Image


class Slide:
    EXT = "svs"
    THUMB_EXT = "png"
    def __init__(self, slide_dir, thumbnail_dir, mask_dir):
        self.slide_dir = slide_dir
        self.thumbnail_dir = thumbnail_dir
        self.mask_dir = mask_dir

    def get_thumbnail(self, slide):
        thumbnail_path = os.path.join(self.thumbnail_dir, f"{slide}_thumb.{self.THUMB_EXT}")
        thumbnail = Image.open(thumbnail_path)
        return thumbnail

    def get_mask(self, slide):
        mask_path = os.path.join(self.mask_dir, f"{slide}_mask.{self.THUMB_EXT}")
        mask = Image.open(mask_path)
        return mask


class SlideProcessor:

    def __init__(self, thumbnail):
        self.thumbnail = np.array(thumbnail, dtype=np.uint8)

        # parameters
        self.blur_kernel = (5, 5)
        self.blur_sigma = 0
        self.opening_kernel = np.ones((5, 5), np.uint8)
        self.closing_kernel = np.ones((5, 5), np.uint8)

        self.area_threshold = 0.99
        self.selection_threshold = 0.01

        # pipeline
        self.mask = None
        self.thresholding()
        self.mask2 = self.select_region()


    def thresholding(self):
        hsv = cv2.cvtColor(self.thumbnail, cv2.COLOR_RGB2HSV)
        _, s, _ = cv2.split(hsv)
        blurred = cv2.GaussianBlur(s, self.blur_kernel, self.blur_sigma)
        _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, self.opening_kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closing_kernel)
        self.mask = closing

    def select_region(self):
        # First pass - approximate objects with overlapping rectangles
        regions = self._select_region(self.mask, threshold=self.area_threshold)

        # Second pass - keep only biggest connected object
        regions = self._select_region(regions, threshold=self.selection_threshold)  # only keep largest region
        return regions

    def _select_region(self, target, threshold):
        contours, hierarchy = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        max_area = max(cv2.contourArea(c) for c in contours)

        regions = np.zeros_like(target, dtype=np.uint8)
        for contour in contours:
            area = cv2.contourArea(contour)
            if (max_area - area) / max_area > threshold:
                continue
            rect = cv2.minAreaRect(contour)
            box = (cv2.boxPoints(rect)).reshape(1, -1, 2).astype(dtype=np.int32)
            cv2.fillPoly(regions, pts=box, color=255)
        return regions

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    samples = [
        "TCGA-AG-3587-01Z-00-DX1.0be1e75e-2c95-406e-bece-b9f5e971be80",
        "TCGA-AA-A029-01Z-00-DX1.36BA3129-431D-4AE5-98E6-BA064D0B5062",
        "TCGA-AA-A02W-01Z-00-DX1.3D9DD408-C389-411D-B4AC-6DC531D35BAD"
    ]

    root_dir = os.path.join(os.path.sep, "home", "felipe", "ExternalDrives")
    slide_dir = os.path.join(root_dir, "+data")
    thumbnail_dir = os.path.join(root_dir, "TCGA_CRC_thumbs", "thumbnails")
    mask_dir = os.path.join(root_dir, "TCGA_CRC_thumbs", "mask")


    for sample in samples:
        slide = Slide(slide_dir, thumbnail_dir, mask_dir)

        mask = slide.get_mask(sample)
        mask_arr = np.array(mask)

        thumbnail = slide.get_thumbnail(sample)
        thumbnail_arr = np.array(thumbnail)

        proc = SlideProcessor(thumbnail)

        test = proc.mask.copy()
        test2 = proc.mask2.copy()


        plt.imshow(thumbnail_arr)
        plt.show()

        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(mask_arr, cmap="gray")
        axs[1].imshow(test, cmap="gray")
        axs[2].imshow(test2, cmap="gray")
        axs[3].imshow(test * test2, cmap="gray")

        fig.tight_layout()
        plt.show()







