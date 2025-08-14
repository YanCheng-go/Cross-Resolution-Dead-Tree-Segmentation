import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.data.base_dataset import BaseDataset


def get_driver_from_type(file_type):
    if 'tif' in file_type:
        return 'GTiff'
    else:
        raise NotImplementedError


class RasterPredictionWriter:
    def __init__(self, test_dataset: BaseDataset, output_path: str = './output', output_prefix: str = 'det_',
                 output_class_labels=False, compression=None, n_edge_removal: int = 0) -> None:
        self.test_dataset = test_dataset
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_class_labels = output_class_labels
        self.output_prefix = output_prefix
        self.cache_image = None
        self.cache_image_path = None
        self.compression = compression
        self.n_edge_removal = n_edge_removal
        if output_class_labels:
            self.dtype = rasterio.int8
        else:
            self.dtype = rasterio.float32

    def dump_cache(self):
        img_name = self.cache_image_path.stem
        op = self.output_path / f"{self.output_prefix}{img_name}.tif"

        ori_img = rasterio.open(self.cache_image_path)
        transform = ori_img.transform
        crs = ori_img.crs

        ch, h, w = self.cache_image.shape

        if not os.path.isfile(op):
            md = 'w'
        else:
            md = 'r+'
            with rasterio.open(op) as pi:
                pi_r = pi.read()
                self.cache_image = np.maximum(self.cache_image, pi_r)
        with rasterio.open(
                op,
                md,
                driver='GTiff',
                width=w,
                height=h,
                count=ch,
                TILED='YES',
                BIGTIFF='IF_SAFER',
                multithread=True,
                NUM_THREADS=8,  # 'ALL_CPUS',
                transform=transform,
                crs=crs,
                nodata=-1,
                compress=self.compression,
                dtype=self.dtype) as dst:
            dst.write(self.cache_image)

        self.cache_image = None
        self.cache_image_path = None

    def init_cache_image(self, img_path: (Path, str), channels: int):
        self.cache_image_path = img_path
        oim = rasterio.open(img_path)
        hm = oim.height
        wm = oim.width
        oim.close()
        # -1 is the nan class
        self.cache_image = np.full(
            (channels, hm, wm), -1 if self.dtype == rasterio.int8 else np.nan, dtype=self.dtype
        )

    def store_in_image_cache(
            self, pred, img_path: (Path, str), col: int, row: int, width: int, height: int, channels: int = None
    ):
        if self.cache_image is not None and img_path != self.cache_image_path:  # Not same as image we were working with
            self.dump_cache()

        if self.cache_image is None:
            self.init_cache_image(img_path, channels)

        _, hm, wm = self.cache_image.shape

        assert len(pred.shape) == 3

        # if the prediction size is not the same as the input patch, we assume a centered (unpadded) prediction
        p_width = pred.shape[2]
        if p_width != width:
            off_width_l = round((width - p_width) / 2)
            off_width_r = (width - p_width) // 2
        else:
            off_width_l = off_width_r = 0

        p_height = pred.shape[1]
        if p_height != height:
            off_height_l = round((height - p_height) / 2)
            off_height_r = (height - p_height) // 2
        else:
            off_height_l = off_height_r = 0

        st_r = max(0, row) + off_height_l
        st_r_pred = 0 if row >= 0 else abs(row)
        st_r_pred += 0 if st_r == 0 else self.n_edge_removal

        end_r = min(hm, row + height) - off_height_r
        end_r_pred = (end_r - st_r) if row + height > hm else height
        end_r_pred += 0 if end_r == hm else -self.n_edge_removal

        st_r += 0 if st_r == 0 else self.n_edge_removal
        end_r -= 0 if end_r == hm else self.n_edge_removal

        st_c = max(0, col) + off_width_l
        st_c_pred = 0 if col >= 0 else abs(col)
        st_c_pred += 0 if st_c == 0 else self.n_edge_removal

        end_c = min(wm, col + width) - off_width_r
        end_c_pred = (end_c - st_c) if col + width > wm else width
        end_c_pred += 0 if end_c == wm else -self.n_edge_removal

        st_c += 0 if st_c == 0 else self.n_edge_removal
        end_c -= 0 if end_c == wm else self.n_edge_removal

        # only save if not over the edge
        if (end_r_pred - st_r_pred) > 0 and (end_c_pred - st_c_pred) > 0:
            self.cache_image[:, st_r: end_r, st_c: end_c] = pred[:, st_r_pred: end_r_pred, st_c_pred: end_c_pred]

    def get_image_path(self, i):
        imgs = self.test_dataset.get_patch_overlapping_images(i)
        # if we are already writing an image, continue with it
        if str(self.cache_image_path) in imgs["path"].values:
            return self.cache_image_path

        return Path(imgs["path"].values[0])

    def __call__(self, pred: dict, dtype=np.float32, *args: Any, **kwds: Any) -> Any:
        # Write the batch prediction
        # Get directory where information should be written
        # Get the file name where image should be written
        # Get the numbers of bands to write
        # Check whether to write raw mask or output

        predictions = pred['predictions']

        ch = predictions.shape[1]

        for i, patch_idx in enumerate(pred['patch_id']):
            # Read only the meta information about the patch
            patch = self.test_dataset.patch_df.loc[patch_idx]

            img_path = self.get_image_path(patch)
            pred_image = pred['predictions'][i]

            col, row, width, height = patch['wn_ori']
            self.store_in_image_cache(pred_image, img_path, col, row, width, height, channels=ch)
