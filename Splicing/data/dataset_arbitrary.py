"""
Updated by Dimitrios Karageorgiou
dkarageo@iti.gr
Mar 31, 2024

Created by Myung-Joon Kwon
mjkwon2021@gmail.com
Sep 10, 2020
"""

import os
from pathlib import Path
from typing import Optional

from PIL import Image

from lib.utils import csv_utils
from Splicing.data.AbstractDataset import AbstractDataset


class ArbitraryDataset(AbstractDataset):
    def __init__(
        self,
        crop_size,
        grid_crop,
        blocks: list,
        dct_channels: int,
        images_source: Path,
        read_from_jpeg=False,
        csv_root: Optional[Path] = None
    ):
        """
        :param crop_size: (H,W) or None
        :param grid_crop:
        :param blocks:
        :param dct_channels:
        :param images_source: Path to a directory or to a dataset's CSV file.
        :param read_from_jpeg: F=from original extension, T=from jpeg compressed image
        :param csv_root: An optional parameter that defines the directory to which the paths
            in the CSV file are relative to. This
        """
        super().__init__(crop_size, grid_crop, blocks, dct_channels)

        self.read_from_jpeg = read_from_jpeg

        if images_source.is_dir():
            self.tamp_list: list[Path] = [f for f in images_source.iterdir() if f.is_file()]
        else:
            manipulated_images: list[Path]
            authentic_images: list[Path]
            manipulated_images, authentic_images, _ = csv_utils.load_dataset_from_csv(
                images_source, csv_root
            )
            self.tamp_list: list[Path] = manipulated_images
            manipulated_images.extend(authentic_images)

    def get_tamp(self, index: int):
        assert 0 <= index < len(self.tamp_list), f"Index {index} is not available!"
        tamp_path: Path = self.tamp_list[index]
        im = Image.open(tamp_path)
        if im.format != "JPEG":
            temp_jpg = f"____temp_{index:04d}.jpg"
            Image.open(tamp_path).convert('RGB').save(temp_jpg, quality=100, subsampling=0)
            tensors = self._create_tensor(temp_jpg, None)
            os.remove(temp_jpg)
        else:
            tensors = self._create_tensor(str(tamp_path), None)
        return tensors
