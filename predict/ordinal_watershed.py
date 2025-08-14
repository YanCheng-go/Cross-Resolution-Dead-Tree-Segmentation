import logging
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import shapes
from shapely.geometry import Polygon, Point
from skimage.segmentation import watershed
from tqdm import tqdm

from predict.segmentation import SegmentationPredictor
from src.data.base_dataset import get_dataloader
from src.data.collate import default_collate_with_shapely_support
from train.ordinal_watershed import generate_mp, mp_func
from train.ordinal_watershed import viz_batch


def get_min_width_height_lenght(geom):
    box = geom.minimum_rotated_rectangle

    # get coordinates of polygon vertices
    x, y = box.exterior.coords.xy

    # get length of bounding box edges
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))

    # get length of polygon as the longest edge of the bounding box
    length = max(edge_length)

    # get width of polygon as the shortest edge of the bounding box
    width = min(edge_length)
    return [length, width]


class OrdinalWatershedPredictor(SegmentationPredictor):
    def __init__(self, config, trainer_cls=None):
        super(OrdinalWatershedPredictor, self).__init__(config, trainer_cls)
        self.mp_dict = generate_mp(config.fuse_mi, config.fuse_mx, config.fuse_interval)
        self.f = lambda x: mp_func(x, self.mp_dict)
        self.use_augmentation = False

    def predict(self, out_path: [str, Path]):

        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True)
        raster_path = out_path / 'rasters'
        raster_path.mkdir(exist_ok=True)
        vector_path = out_path / 'vectors'
        vector_path.mkdir(exist_ok=True)

        if os.listdir(out_path):
            logging.warning(f"Directory is not empty! {out_path}: {os.listdir(out_path)}")

        # go through images
        areas = self.idf.query(f"src == @self.config.reference_source")[['geometry']].copy()
        # predict each image individually
        for i in tqdm(areas.index.values, f"Predicting tile ..."):
            image_df = self.idf.query(f"src != @self.config.reference_source or index == @i").copy()
            area = areas.loc[[i]].copy()
            project_crs = self.idf.loc[i]["ori_crs"]
            if image_df.crs != project_crs:
                image_df = image_df.to_crs(project_crs)
                area = area.to_crs(project_crs)
            test_dataset = self.dataset_cls(
                labeled_areas=area, images_df=image_df,
                reference_source=self.config.reference_source,
                target_features=None,
                target_col=None,
                target_classes=None,
                patch_size=self.config.patch_size,
                sequential_stride=self.config.sequential_stride,
                allow_empty_patches=True,
                allow_partial_patches=self.config.allow_partial_patches,
                processed_dir=self.config.processed_dir, save_samples=self.config.save_samples,
                save_patch_df=self.config.save_patch_df,
                save_labeled_areas_df=self.config.save_labeled_areas_df,
                project_crs=project_crs
            )

            test_image_loader = get_dataloader(
                test_dataset, self.config.batch_size_val, self.config.num_workers,
                collate_fn=default_collate_with_shapely_support, train=False
            )

            """Evaluation of the network"""
            self.predict_image(out_path, test_image_loader)

    def predict_image(self, out_path, test_image_loader):
        output_class_labels = self.config.get("output_class_labels", True)
        do_vectorize = self.config.get("do_vectorize", True)
        do_write_raster = self.config.get("do_write_raster", True)
        assert not do_vectorize or do_vectorize == output_class_labels, "If you want to vectorize, enable output_class_labels"
        if output_class_labels:
            print("Warning: this script only gives watershed levels and instance output, no direct class labels")

        output_prefix = 'det_'
        pred_writer = self.raster_writer_cls(
            test_image_loader.dataset, Path(out_path) / 'rasters', output_prefix, False, self.config.get("compression", None),
            self.config.n_edge_removal
        )
        n_batches = len(test_image_loader)
        self.model.eval()
        self.aug_transform.eval()
        with tqdm(total=n_batches, desc=f'Evaluating ...', leave=False) as pbar:
            with torch.no_grad():
                for batch in test_image_loader:
                    x = self.aug_transform.get_input(batch, self.device)  # Normalize

                    patch_res = torch.tensor(round(batch["ori_transform"][0][0], 2))
                    additional_scalars = [patch_res.to(self.device)]
                    # viz_batch(x_batch=x, y_batch=y_energy, batch_id=i)
                    if 'with_scalar' in self.config.model_type:
                        pred_sobel, pred_energy = self.model(x, additional_scalars)
                    else:
                        pred_sobel, pred_energy = self.model(x)

                    pred_energy.masked_fill_((x == 0).all(1, keepdim=True), 0)  # ensuring that outside areas are empty
                    pred = (pred_energy > 0).float().cumprod(1).sum(1, keepdims=True)

                    if self.config.fuse_out is True:
                        pred = torch.from_numpy(np.vectorize(self.f)(pred.cpu())).to(self.device)

                    prb = {
                        'patch_id': batch['patch_id'].cpu().numpy(),
                        'predictions': pred.cpu().numpy(),
                    }
                    pred_writer(prb)
                    pbar.update()
        if output_class_labels:
            pred_writer.cache_image[np.isnan(pred_writer.cache_image)] = -1
            energy = pred_writer.cache_image[0]
            mask = pred_writer.cache_image[0] > 0
            pred_watershed = watershed(-energy, connectivity=2, mask=mask)

            if do_vectorize and pred_watershed.sum() > 0:
                # vectorize instances
                ori_img = rasterio.open(pred_writer.cache_image_path)
                transform = ori_img.transform
                crs = ori_img.crs
                poly = shapes(
                    pred_watershed.astype(np.int32), mask=pred_watershed != 0, transform=transform, connectivity=8
                )

                poly = gpd.GeoDataFrame(geometry=[Polygon(p[0]["coordinates"][0]) for p in poly], crs=crs)
                w_h = np.sort(np.array([get_min_width_height_lenght(geom) for geom in poly.geometry.values]))
                poly["shortest_edge"], poly["longest_edge"] = w_h.T
                poly.eval("edge_difference = longest_edge - shortest_edge", inplace=True)
                poly["area"] = poly.area
                poly.to_file(f"{out_path}/vectors/{output_prefix}{pred_writer.cache_image_path.stem}.gpkg")

            pred_writer.cache_image = np.stack([energy, pred_watershed], 0)
            pred_writer.cache_image = pred_writer.cache_image.astype(int)
            pred_writer.dtype = rasterio.int32

        if do_write_raster:
            pred_writer.dump_cache()
