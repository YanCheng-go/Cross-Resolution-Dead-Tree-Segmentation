import logging
import math
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
import torch
from pyproj import Transformer
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import MultiPolygon
from shapely.ops import transform
from shapely.ops import unary_union
from torch.utils.data import Dataset, Subset

from config.base import BaseConfig


def draw_energy_masks_with_levels(mask):
    """{k: list(v) for k, v in df.groupby('Column1')['Column3']}areasWithPolygons
    From the polygons, create a numpy mask with fill value in the foreground and 0 value in the background.
    Outline (i.e the edge of the polygon) can be assigned a separate value.
    """
    lvl = 1
    erosion = mask.copy()
    while erosion.sum() > 0:  # Until we erode the whole image
        kernel = np.ones((3, 3), np.uint8)  # Erode by 2 pixels
        erosion = cv2.erode(erosion, kernel, iterations=1)
        er = erosion.copy() * lvl
        mask = np.maximum.reduce([mask, er])
        lvl += 1
    return mask


def get_centered_aoi_mask(patch_shape, aoi, expand_dims_axis=None):
    px, py = patch_shape
    ax, ay = aoi
    dx = int((px - ax) / 2)
    dy = int((py - ay) / 2)
    aoi_mask = np.zeros(patch_shape)
    aoi_mask[dx: dx + ax, dy: dy + ay] = 1
    if expand_dims_axis is None:
        return aoi_mask
    else:
        return np.expand_dims(aoi_mask, axis=expand_dims_axis)


def get_polygons_in_patch(polygons_areas_table, patch_geom):
    l = []
    for tp in polygons_areas_table:
        if patch_geom.intersects(tp):
            l.append(tp)
    return MultiPolygon(l)


def extract_labeled_areas_from_patch_dict(patch_df_path, labeled_areas_df_path):
    patch_dict = torch.load(patch_df_path)
    if "assigned_images" in patch_dict and "labeled_areas" in patch_dict:
        # Rewrite labeled dict
        assigned_images = patch_dict["assigned_images"]
        labeled_areas = patch_dict["labeled_areas"]
        lb_dict = {
            "labeled_areas": labeled_areas,
            "assigned_images": assigned_images,
        }
        torch.save(lb_dict, labeled_areas_df_path)

        # Rewrite patch dict
        patch_df = patch_dict["patch_df"]
        patch_dict = {
            "patch_df": patch_df,
        }
        torch.save(patch_dict, patch_df_path)


class BaseDataset(Dataset):
    """
    RemoteSensingDataset is designed to handle multiple sources of images at once.
    These sources may have different resolutions, channels or CRS.
    And the dataset would return a sample image accordingly. However, this flexibility has some limitations.
    For example, the expected size in different resolution maybe a couple of bit off,
    (assuming that they perfectly overlap).
    More likely, a sample may lie across multiple tiles in some sources,
    for that matter mapping each point on earth is still a very challenging task and it has its limitations.
    While we find work around these limitations, it may not be possible always.

    Note, that all the shape of patch, aoi and stride are given specific to a reference image source
    (assuming there are multiple crs), and rest are of the shapes are sizes are calculated proportionally.
    """

    def __init__(
            self,
            labeled_areas: gpd.GeoDataFrame,
            images_df: gpd.GeoDataFrame,
            reference_source: str,
            patch_size=(512, 512),
            allow_partial_patches: bool = False,
            sequential_stride=(256, 256),
            dataset_transform=None,
            processed_dir: [str, Path] = None,
            save_samples: bool = False,
            save_patch_df: bool = False,
            save_labeled_areas_df: bool = False,
            project_crs: str = None,
            dataset_name: str = 'train',
    ) -> None:
        super(Dataset, self).__init__()
        self.images_df = images_df
        self.reference_source = str(reference_source)
        assert patch_size is not None
        if type(patch_size) == int:
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.allow_partial_patches = allow_partial_patches
        assert sequential_stride is not None
        if type(sequential_stride) == int:
            sequential_stride = (sequential_stride, sequential_stride)
        self.sequential_stride = sequential_stride
        assert project_crs is not None
        self.project_crs = project_crs
        self.dataset_name = dataset_name
        if processed_dir:
            logging.info(f"Processed_dir path: {processed_dir}")
            processed_dir = Path(processed_dir)
            # create processed sample dir
            processed_dir.mkdir(exist_ok=True)
            patch_df_path = processed_dir / f"{dataset_name}_patch_df.pt"
            labeled_areas_df_path = processed_dir / f"{dataset_name}_labeled_areas_df.pt"
        self.processed_dir = processed_dir
        self.save_samples = save_samples
        self.save_patch_df = save_patch_df
        self.save_labeled_areas_df = save_labeled_areas_df
        if (save_samples or save_patch_df or save_labeled_areas_df) and not self.processed_dir:
            raise Exception("No 'processed_dir' provided, cannot save samples or patch_df or save_labeled_areas_df.")
        elif not (save_samples or save_patch_df or save_labeled_areas_df) and self.processed_dir:
            warnings.warn("You provided 'processed_dir' provided but neither samples nor patch_df are saved.")
        elif processed_dir:
            (processed_dir / 'patches').mkdir(exist_ok=True)
            logging.info(f"Saving samples:{save_samples}, patch_df: {save_patch_df} to {self.processed_dir}")

        self.dataset_transform = dataset_transform

        # For backward compatibility; it can be removed once it has been run for all training scripts.
        if self.processed_dir and self.save_labeled_areas_df and self.save_patch_df and patch_df_path.exists():
            extract_labeled_areas_from_patch_dict(patch_df_path, labeled_areas_df_path)

        if self.processed_dir and self.save_labeled_areas_df and labeled_areas_df_path.exists():
            logging.info(f"Reading labeled areas from {labeled_areas_df_path}")
            lb_dict = torch.load(labeled_areas_df_path)
            self.labeled_areas = lb_dict["labeled_areas"]
            self.assigned_images = lb_dict["assigned_images"]
            logging.info(f"Done reading labeled areas")
        else:
            logging.info(f"Creating labeled areas and assigned images for dataset {self.dataset_name}")
            self.labeled_areas, self.assigned_images = self.build_labeled_areas_table(labeled_areas.copy())
            if self.processed_dir and self.save_labeled_areas_df:
                lb_dict = {
                    "labeled_areas": self.labeled_areas,
                    "assigned_images": self.assigned_images,
                }
                torch.save(lb_dict, labeled_areas_df_path)

        if self.processed_dir and self.save_patch_df and patch_df_path.exists():
            patch_dict = torch.load(patch_df_path)
            self.patch_df = patch_dict["patch_df"]
            self.length = len(self.patch_df)
        else:
            self.patch_df, self.length = self.build_patch_table_sequentially()
            if self.processed_dir and self.save_patch_df:
                patch_dict = {"patch_df": self.patch_df}
                torch.save(patch_dict, patch_df_path)

    # override this
    def get_patch(self, idx):
        raise NotImplementedError()

    # override this, very likely can use self.build_labeled_areas_table_ but needs to process targets somehow
    def build_labeled_areas_table(self, labeled_areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        raise NotImplementedError()

    # override this
    def build_patch_table_sequentially(self, patch_start_index=0):
        raise NotImplementedError()

    def __len__(self):
        return self.length

    def get_patch_overlapping_images(self, patch):
        ov_im = self.assigned_images.query("area_id == @patch.area_id")
        # For multiple sources, creates patches on the reference source and then select images that overlap the patch
        overlapping_images = self.images_df.loc[ov_im.reset_index().image_id]
        overlapping_images = overlapping_images[overlapping_images.intersects(patch.geometry)]
        return overlapping_images

    def get_patch_raster_images(self, patch):
        pt = {}
        overlapping_images = self.get_patch_overlapping_images(patch)
        # Handle different source by defining window using bounds
        # While it's nice to have overlapping images in the patch sample, it creates problems in dataloader collate.
        # pt['overlapping_images'] = overlapping_images.index.tolist()

        for _, img in overlapping_images.iterrows():
            src_name = img["src"]
            # transform to reference image crs
            patch_geom = patch.geometry
            wn_geom = geometry_to_crs(patch_geom, self.project_crs, img["ori_crs"])
            # careful here, the windows is from the bounds not the geometry of the patch
            wn = rasterio.windows.from_bounds(*wn_geom.bounds, transform=img['ori_transform'])
            # This is required due to a rounding error while creating the window and case when the window is less than 1 pixel
            wn = Window(np.floor(wn.col_off), np.floor(wn.row_off), max(1, round(wn.width)), max(1, round(wn.height)))

            img_arr = self.read_window_from_image(img_path=img["path"], wn=wn, boundless=True)
            # need to reproject if different reference crs was used
            if img["ori_crs"] != patch["ori_crs"] or img_arr.shape[1:] != patch["shape"]:
                # get transform and shape of reprojection
                patch_res = (patch["ori_transform"].a, -patch["ori_transform"].e)
                img_res = (img["ori_transform"].a, -img["ori_transform"].e)
                # only override resolution if necessary
                resolution = None if np.isclose(patch_res, img_res).all() else patch_res

                if img["ori_crs"] != patch["ori_crs"]:  # if different crs/resolution
                    transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                        img["ori_crs"], patch["ori_crs"], *img_arr.shape[1:], *wn_geom.bounds,
                        resolution=resolution
                    )
                else:  # if only different resolution
                    transform, dst_width, dst_height = rasterio.warp.aligned_target(
                        img["ori_transform"], *img_arr.shape[1:], resolution)

                # just adapt transform
                dst = np.zeros((img_arr.shape[0], dst_width, dst_height), dtype=img_arr.dtype)
                dst, _ = rasterio.warp.reproject(
                    source=img_arr,
                    destination=dst,
                    src_transform=img["ori_transform"],
                    src_crs=img["ori_crs"],
                    dst_transform=transform,
                    dst_crs=patch["ori_crs"],
                    resampling=Resampling.nearest
                )
                # center image to same size
                dst_x_off = dst.shape[1] - patch["shape"][0]
                dst_y_off = dst.shape[2] - patch["shape"][1]

                # only crop if there is a shape difference
                if dst_x_off == 0:
                    dst_x_off_max = dst_x_off_min = None
                else:
                    dst_x_off_min = dst_x_off // 2
                    dst_x_off_max = -int(np.ceil(dst_x_off / 2))

                if dst_y_off == 0:
                    dst_y_off_max = dst_y_off_min = None
                else:
                    dst_y_off_min = dst_y_off // 2
                    dst_y_off_max = -int(np.ceil(dst_y_off / 2))

                dst = dst[:, dst_x_off_min: dst_x_off_max, dst_y_off_min: dst_y_off_max]
                img_arr = dst

            img_arr = img_arr.astype(np.float32)
            nan_mask = np.isnan(img_arr) | (img_arr == img["nodatavals"])
            img_arr[nan_mask] = np.nan
            if src_name in pt:
                if self.merge_mode == "keep_first":
                    nan_mask = np.isnan(pt[src_name])
                    pt[src_name] = np.where(nan_mask, img_arr, pt[src_name])
                if self.merge_mode == 'keep_last':
                    pt[src_name] = np.where(nan_mask, pt[src_name], img_arr)
                if self.merge_mode == 'keep_biggest': # for patch grid that are located on the edge of one tile
                    first_mask = np.isnan(pt[src_name]) | (pt[src_name] == 0)
                    last_mask = np.isnan(img_arr) | (img_arr == img["nodatavals"]) | (img_arr == 0)
                    if first_mask.sum() <= last_mask.sum():
                        pt[src_name] = np.where(nan_mask, img_arr, pt[src_name])
                    else:
                        pt[src_name] = np.where(nan_mask, pt[src_name], img_arr)
            else:
                pt[src_name] = img_arr
        for src_name in overlapping_images["src"].unique():
            pt[src_name] = torch.tensor(pt[src_name], dtype=torch.float32)

        return pt

    def build_labeled_areas_table_(self, labeled_areas):
        """
        Index, src, geometry
        ["image_id", "area_id"], 'dg_50', 'Polygon()'
        """
        # create a lookup table (already taking the needed infos (needs more ram but is ok))
        assigned_images = gpd.sjoin(
            self.images_df[["src", "geometry"]], labeled_areas[["geometry"]],
            predicate="intersects", how="inner"
        ).rename(columns={"index_right": "area_id"})
        # filter out not found values
        missing_areas = labeled_areas.query("index not in @assigned_images.area_id")
        labeled_areas = labeled_areas.query("index in @assigned_images.area_id")
        if len(missing_areas) > 0:
            logging.info(f"{len(missing_areas)} areas with no image source, idx: {missing_areas.index}")
            logging.info(f"ignoring them for now")
        complete_areas = assigned_images.query("src == @self.reference_source").area_id
        missing_areas = labeled_areas.query("index not in @complete_areas")
        labeled_areas = labeled_areas.query("index in @complete_areas")
        assigned_images = assigned_images.query("area_id in @complete_areas")
        if len(missing_areas) > 0:
            logging.info(f"{len(missing_areas)} areas with no input image source, idx: {missing_areas.index}")
            logging.info(f"ignoring them for now")
        assigned_images.set_index(['area_id'], append=True, inplace=True)
        assert assigned_images.index.names == ["image_id", "area_id"]
        # assigned_images.set_index(["area_id", "image_id"], inplace=True)
        area_info = []
        for area_id, group_df in assigned_images.groupby("area_id"):
            area_geom = labeled_areas.loc[area_id]['geometry']

            def get_overlapping_area(row):
                # These transform represent the overlapping transform, shape and geometry so renamed it to ov from area
                return pd.Series(
                    [area_geom.intersection(row['geometry'])],
                    index=["ov_geometry"])

            overlapping_area = group_df.apply(get_overlapping_area, axis=1)
            area_info.append((overlapping_area))

        area_info = pd.concat(area_info, axis=0)
        # add to lookup table
        assigned_images = assigned_images.join(area_info)
        # Geometry and transform refer to the geometry of the original image instead of its overlap with the rectangle.
        # So we delete these columns and replace them with the common area-image transform and geometry
        assigned_images.set_geometry('ov_geometry', inplace=True)
        assigned_images.drop(labels=['geometry'], axis="columns", inplace=True)
        assigned_images.rename_geometry('geometry', inplace=True)

        return labeled_areas, assigned_images

    def get_imgs_reference(self, imgs):
        # if more than one image from the reference source is available for the patch, then create a virtual raster
        if len(imgs) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                idx_largest_overlap = np.argmax(imgs.area)
            union_geometry = unary_union(imgs.geometry.to_list())
            ori_transform = imgs["ori_transform"].iloc[idx_largest_overlap]
            ori_crs = imgs["ori_crs"].iloc[idx_largest_overlap]
        else:
            union_geometry = imgs["geometry"].iloc[0]
            ori_transform = imgs["ori_transform"].iloc[0]
            ori_crs = imgs["ori_crs"].iloc[0]
        return ori_crs, ori_transform, union_geometry

    def is_complete(self, row, image_source: str):
        if image_source in row["overlapping_images"]:
            return True
        return False

    def get_image_row(self, image_id):
        return self.images_df.query("image_id == @image_id").iloc[0]

    def get_rows_cols_from_geom(self, target_crs, ori_transform, geometry):
        minx, miny, maxx, maxy = geometry.bounds
        transformer_to_ori = Transformer.from_crs(self.project_crs, target_crs, always_xy=True)
        # Get the points in image coordinate reference system
        ic_minx, ic_maxy = transformer_to_ori.transform(minx, maxy)  # Top left
        ic_maxx, ic_miny = transformer_to_ori.transform(maxx, miny)  # Bottom right
        # In a normal array, row, col (0,0) means top left and row increase downwards
        #   while the column increase right wards
        # In a geo reference image with xy (longitude, latitude) orientation,
        #   y decreases south wards (opposite of rows) and x increases east wards (same as columns)
        row_start, col_start = rasterio.transform.rowcol(
            ori_transform, ic_minx, ic_maxy)  # Bottom left, i.e image start
        row_end, col_end = rasterio.transform.rowcol(
            ori_transform, ic_maxx, ic_miny)  # Top right, i.e image end
        return col_start, col_end, row_start, row_end

    def __getitem__(self, idx):
        if self.processed_dir and self.save_samples:
            path = self.processed_dir / 'patches' / f"{self.dataset_name}_{idx}.pt"
            if path.exists():
                try:
                    sample = torch.load(path)
                except Exception as e:
                    logging.info(e)
                    sample = self.get_patch(idx)
                    torch.save(sample, path)
            else:
                sample = self.get_patch(idx)
                torch.save(sample, path)
        else:
            sample = self.get_patch(idx)
        if self.dataset_transform:
            sample = self.dataset_transform(sample)
        return sample

    @classmethod
    def init_from_config(self, config: BaseConfig, dataset_name: str, labeled_areas: gpd.GeoDataFrame,
                         images_df: gpd.GeoDataFrame, dataset_transform=None, **kwargs) -> Tuple[dict, BaseConfig]:
        raise NotImplementedError("function init_from_config is not implemented")

    # def init_data(self, config: BaseConfig) -> Tuple[dict, BaseConfig]:
    #     """load idf, areas, and maybe polygons as dictionary"""
    #     raise NotImplementedError("function init_data is not implemented")

    @staticmethod
    def save(dataset, path):
        if not str(path).endswith('.pt'):
            path = path / f'dataset.pt'
        try:
            torch.save(dataset, path)
            return 1
        except Exception as e:
            return 0

    @staticmethod
    def load(path):
        if not str(path).endswith('.pt'):
            path = path / f'dataset.pt'
        return torch.load(path)

    @staticmethod
    def patches_count_calculator(area_dim, patch_dim, stride_dim, first_partial_patch=True):
        n = ((area_dim - patch_dim) / stride_dim) + 1
        if first_partial_patch:
            return max(1, math.ceil(n))
        else:
            return max(0, math.floor(n))

    @staticmethod
    def read_window_from_image(img_path, wn, boundless=True, fill_value=0):
        with rasterio.open(img_path) as src:
            rw = src.read(window=wn, boundless=boundless, fill_value=fill_value)
        return rw


def train_val_split_data(val_split, train_set, val_transform, rnds=42):
    val_set = deepcopy(train_set)
    if isinstance(val_set, BaseDataset):
        val_set.dataset_transform = val_transform
    else:
        try:
            # try to reach the basedataset
            tmp_set = val_set.dataset
            while not isinstance(tmp_set, BaseDataset):
                tmp_set = val_set.dataset
            tmp_set.dataset_transform = val_transform
        except AttributeError:
            logging.error("Given dataset not inherited from BaseDataset, cannot set val_transform")

    if val_split > 1 - val_split:
        logging.error(
            f'Inside train_val_split_data. '
            f'The val_split ({val_split}) is bigger than the train_split ({1 - val_split})! '
            f'Make sure this is intended. '
        )

    if val_split is None or val_split == 1:
        train_subset, val_subset = train_set, val_set
    else:
        num_train = len(train_set)
        indices = np.arange(num_train)
        rs = np.random.RandomState(rnds)
        rs.shuffle(indices)
        val_split = int(np.floor((1 - val_split) * num_train))
        train_idx, val_idx = indices[:val_split], indices[val_split:]
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)
    return train_subset, val_subset


def splits_from_array(splits, dataset, transforms: dict):
    """
    params:
    splits: array indicating split per sample (e.g. 0 = train, 1 = test), needs to have the same length as train_set
    dataset: Dataset object to split
    transforms: dictionary mapping split value to a transform (e.g. { 0: train_transform, 1: test_transform})

    return:
    dict mapping split value to dataset
    """
    num_samples = len(dataset)
    split_values = np.unique(splits)
    indices = np.arange(num_samples)

    split_dict = {}
    for split_value in split_values:
        transform = transforms.get(split_value, None)
        split_indices = indices[splits == split_value]
        subset = deepcopy(dataset)
        subset.dataset_transform = transform
        subset = Subset(subset, split_indices)
        split_dict[split_value] = subset

    return split_dict


def train_val_test_split_data(val_split, test_split, train_set, val_transform, test_transform, rnds=42):
    rs = np.random.RandomState(rnds)

    # create unit_test split
    if (test_split > 1 - test_split - val_split):
        logging.error(
            f'The val_split ({test_split}) is bigger than the train_split '
            f'({1 - test_split - val_split})! Make sure this is intended.')
    test_set = deepcopy(train_set)
    test_set.dataset_transform = test_transform
    num_train = len(train_set)
    indices = np.arange(num_train)
    rs.shuffle(indices)
    split = int(np.floor((1 - test_split) * num_train))
    train_idx, test_idx = indices[:split], indices[split:]
    train_subset = Subset(train_set, train_idx)
    test_subset = Subset(test_set, test_idx)

    # create val split
    if (val_split > 1 - test_split - val_split):
        logging.error(
            f'The val_split ({val_split}) is bigger than the train_split '
            f'({1 - test_split - val_split})! Make sure this is intended.')
    val_set = deepcopy(train_set)
    val_set.dataset_transform = val_transform
    if val_split is None or val_split == 1:  # val split is train set
        val_subset = val_set
    else:
        split = int(np.floor((1 - test_split - val_split) * num_train))
        train_idx, val_idx = train_idx[:split], train_idx[split:]
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(val_set, val_idx)

    return train_subset, val_subset, test_subset


def get_dataloader(dataset, batch_size, num_workers, collate_fn, train: bool, pin_memory=False, sampler=None,
                   shuffle=None):
    if sampler is None:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=train, num_workers=num_workers, pin_memory=pin_memory,
            shuffle=train, collate_fn=collate_fn, sampler=sampler
        )
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=train, num_workers=num_workers, pin_memory=pin_memory,
            shuffle=shuffle, collate_fn=collate_fn, sampler=sampler
        )


def geometry_to_crs(geom, from_crs, to_crs):
    gdf = gpd.GeoDataFrame({'geometry': [geom]}).set_crs(from_crs)
    gdf = gdf.to_crs(to_crs)
    return gdf['geometry'].iloc[0]


def geometry_to_crs_pyproj(geom, from_crs, to_crs):
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(from_crs),  # source coordinate system
        pyproj.Proj(to_crs),
        always_xy=True
    )  # destination coordinate system

    return transform(project.transform, geom)


def transform_to_crs(project_crs='EPSG:4326'):
    def in_new_crs(df):
        ndf = df.set_crs(df['ori_crs'].iloc[0], allow_override=True)
        ndf.to_crs(project_crs, inplace=True)
        return ndf

    return in_new_crs


def get_df_in_single_crs(df, project_crs):
    crs_grouped = df.groupby('ori_crs', sort=False)
    to_new_crs = transform_to_crs(project_crs)
    tdf = crs_grouped.apply(to_new_crs)  # Transform the geometry to the new crs
    return tdf
