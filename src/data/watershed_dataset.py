import functools
import itertools
import math
import os
import typing
import warnings
from copy import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
import torch
from geopandas import GeoDataFrame
from kornia.augmentation import AugmentationSequential, RandomAffine
from kornia.geometry import build_pyramid
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, distance_transform_edt
from shapely.ops import unary_union
from skimage import filters
from tqdm import tqdm

from torchvision import transforms

from src.data.base_dataset import get_df_in_single_crs, geometry_to_crs
from src.data.base_segmentation_dataset import BaseSegmentationDataset, get_patch_features
from src.modelling import helper

import geopandas as gpd
import logging

from config.base import BaseSegmentationConfig

from itertools import product
from rasterio import windows
from shapely.geometry import box
from rasterio.windows import Window
import torch
from typing import Dict, Any, Union

def watershed_mask2(seg_mask: object, out_shape: object) -> object:
    fe = np.zeros(out_shape)
    # We rasterize each polygon independently to ensure a clean edge.
    for i in seg_mask.unique():
        new_mask = seg_mask.clone().detach()
        new_mask[new_mask != i] = 0
        new_mask[new_mask != 0] = 1
        e = distance_transform_edt(new_mask.cpu())
        fe = np.maximum(fe, e)

    sobelx64f = filters.sobel_h(fe)
    sobely64f = filters.sobel_v(fe)
    n = np.sqrt(sobelx64f * sobelx64f + sobely64f * sobely64f)
    n[n == 0] = 1
    sx = (sobelx64f / n)
    sy = (sobely64f / n)
    return np.ceil(fe), sx, sy


def gaussian_density(polygons: GeoDataFrame, out_shape, sigma: float, transform, pad:int=0):
    """
    Create a gaussian density map from a set of polygons.
    If two gaussian kernels overlap, the values are added. If a gaussian kernel is partially outside the image,
     its other values are scaled accordingly. Padding is used to avoid this.
    Unless padding is used, the output always sums up to the total number of points when sigma = 1.
    :param polygons: A GeoDataFrame with a geometry column
    :param out_shape: The shape of the output density map
    :param sigma: The sigma of the gaussian kernel
    :param transform: The transform of the output density map
    :param pad: The number of pixels to pad to the output density map to avoid edge effects. The padding is done symmetrically.
                The default value is 0, which means no padding. Padding is afterwards removed from the output density map.
    :return: A density map
    """
    # first only get center points
    if len(polygons) > 0:
        if set(polygons.geom_type.values) != set(['Point']):
            points = polygons.centroid.to_list()
        else:
            points = polygons.geometry.to_list()
        point_y = rasterio.features.rasterize(
            points, out_shape=out_shape, dtype=float, merge_alg=rasterio.enums.MergeAlg.add,
            all_touched=False, fill=0, transform=transform
        )
        if pad == 0:
            gauss_density = gaussian_filter(point_y, sigma=sigma)
        else:
            point_y = np.pad(point_y, pad, mode='constant', constant_values=0) # Pad to avoid edge effects
            gauss_density = gaussian_filter(point_y, sigma=sigma)
            gauss_density = gauss_density[pad:-pad, pad:-pad] # Remove padding
    else:
        gauss_density = np.zeros(out_shape)

    return gauss_density


def watershed_mask(polys_ori_crs, out_shape, all_touched, fill, transform):
    fe = np.zeros(out_shape)
    # We rasterize each polygon independently to ensure a clean edge.
    for p in polys_ori_crs:
        m = rasterio.features.rasterize(
            [p], out_shape=out_shape,
            all_touched=all_touched, fill=fill, transform=transform
        )
        e = distance_transform_edt(m)
        fe = np.maximum(fe, e)

    sobelx64f = filters.sobel_h(fe)
    sobely64f = filters.sobel_v(fe)
    n = np.sqrt(sobelx64f * sobelx64f + sobely64f * sobely64f)
    n[n == 0] = 1
    sx = (sobelx64f / n)
    sy = (sobely64f / n)
    return np.ceil(fe), sx, sy


def calculate_size_weights(mask: torch.Tensor, small_object_threshold=None, device='cuda') -> torch.Tensor:
    """Add a weight to each object based on the area of each object, i.e. small objects get higher weights.
    If there is only one object within a patch, both the object and the background pixels get a weight of 1.
    If there are multiple objects, the weight is calculated based on the area of the object.
    The weight is higher for smaller objects and lower for larger objects. The highs and lows are defined by
    comparing to the small_object_threshold or a relative size rank within a patch.
    The average weight for each patch is 1."""

    # Move mask to the specified device
    if device is not None:
        mask = mask.to(device)

    # Check if the mask is empty (only background)
    if mask.sum() == 0:
        return torch.ones_like(mask, dtype=torch.float)
    else:
        mask = mask.long()  # Ensure mask is an integer tensor
        unique_objects = mask.unique(sorted=True)  # Get unique objects sorted

        # Calculate the area (count) for each object except for background (assumed to be 0)
        object_areas = {int(obj): (mask == obj).sum().item() for obj in unique_objects if obj != 0}

        # Sort dictionary based on the area (count)
        object_areas = dict(sorted(object_areas.items(), key=lambda item: item[1]))

        # Find the middle index of the count if no threshold is provided
        if small_object_threshold is None:
            middle_index = len(object_areas.values()) // 2 - 1 if len(object_areas.values()) % 2 == 0 else len(object_areas.values()) // 2
            small_object_threshold = list(object_areas.values())[middle_index]

        # Calculate initial weights
        weight_map = torch.ones_like(mask, dtype=torch.float)
        for obj, area in object_areas.items():
            if area < small_object_threshold:
                weight = 1.0 + (small_object_threshold / area)  # Higher weight for smaller objects
            else:
                weight = 1.0 / (area / small_object_threshold)  # Lower weight for larger objects
            weight_map[mask == obj] = weight

        # Normalize the weights for non-zero pixels so the average weight is 1
        non_zero_mask = mask != 0
        normalization_factor = non_zero_mask.sum() / weight_map[non_zero_mask].sum()
        weight_map[non_zero_mask] *= normalization_factor

        return weight_map


def calculate_edge_weights(mask: torch.Tensor, energy_layer: torch.Tensor, edge_weight=10, edge_threshold_pixel=3,
                           edge_threshold_percent=0.2, device='cuda') -> torch.Tensor:
    """Add a weight to each pixel based on the distance to the nearest edge."""

    # Move mask and energy_layer to the specified device
    if device is not None:
        mask = mask.to(device)
        energy_layer = energy_layer.to(device)

    if mask.sum() == 0:
        return torch.ones_like(mask, dtype=torch.float, device=device)

    mask = mask.long()  # Ensure mask is an integer tensor
    unique_objects = torch.unique(mask)  # Get unique objects sorted

    # Filter out the background
    unique_objects = unique_objects[unique_objects != 0]

    # If there are no non-background objects, return a tensor of ones
    if unique_objects.numel() == 0:
        return torch.ones_like(mask, dtype=torch.float, device=device)

    # Find out max energy for each object
    max_energy_dict = {int(obj): energy_layer[mask == obj].max() for obj in unique_objects}
    max_energy = torch.ones_like(mask, dtype=torch.float, device=device)

    for obj in unique_objects:
        max_energy[mask == obj] = max_energy_dict.get(int(obj), 0)

    # Compute edge threshold for each object
    if edge_threshold_percent is not None:
        edge_threshold = torch.maximum(torch.ceil(max_energy * edge_threshold_percent),
                                       torch.tensor(edge_threshold_pixel, dtype=torch.float, device=device))
    else:
        edge_threshold = torch.full_like(max_energy, edge_threshold_pixel, dtype=torch.float)

    # Create a mapping from object id to edge threshold
    object_to_threshold = {int(obj): edge_threshold[mask == obj].max().item() for obj in unique_objects}
    object_to_threshold.update({0: 0})

    # Apply the mapping using vectorized operations
    thresholds = torch.zeros_like(mask, dtype=torch.float, device=device)
    for obj, threshold in object_to_threshold.items():
        thresholds[mask == obj] = threshold

    # Apply the weights
    result = torch.ones_like(energy_layer, dtype=torch.float, device=device)
    result[(mask != 0) & (energy_layer <= thresholds)] = edge_weight

    return result


class WatershedDataset(BaseSegmentationDataset):
    '''
        RemoteSensingDataset is designed to handle multiple sources of images at once.
        These sources may have different resolutions, channels or CRS.
        And the dataset would return a sample image accordingly. However, these flexibility has some limitations.
        For example, the expected size in different resolution maybe a couple of bit off,
        (assuming that the perfectly overlap).
        More likely, a sample may lie across multiple tiles in some sources,
        for that matter mapping each point on earth is still a very challenging task and it has its limitations.
        While we find work around these limitations, it may not be possible always.

        Note, that all the shape of patch, aoi and stride are given specific to a reference image source (assuming there are multiple),
        and rest are of the shapes are sizes are calculated proportionally.
    '''

    def __init__(
            self,
            labeled_areas: gpd.GeoDataFrame,
            images_df: gpd.GeoDataFrame,
            reference_source: str,
            project_crs,
            target_features: Optional[gpd.GeoDataFrame],  # e.g, tree polygons (hand labels).
            target_col: Optional[str] = None,  # e.g 'feature_type', or 'tree_species'
            target_classes: list = None,
            patch_size=(512, 512),
            allow_partial_patches: bool = False,
            allow_empty_patches: bool = True,
            sequential_stride=(256, 256),
            dataset_transform=None,
            processed_dir=None,
            save_samples: bool = False,
            save_patch_df: bool = False,
            save_labeled_areas_df: bool = False,
            extract_images_for_areas: bool = False,
            dataset_name: str = 'train',
            merge_mode: str = "keep_first",
            rasterize_all_touched: bool = False,
            append_data: bool = False,
            year_col=None,
            rescale=False,
            epochs=1,
            weighted_sampling=False,
            dataset_split_by='patches',
            edge_weight=10,
            edge_threshold_pixel=3,
            edge_threshold_percent=0.2,
            small_object_threshold=None,
            device='cuda',
            auto_resample=False,
            res_list=[0.2, 0.3, 0.4, 0.5, 0.6],
            allow_upsample=True,
            res_tolerance=0.02,
            saved_resampled_patches=False,
            max_value=255,
            normalize=True,
            normalize_by_dataset=False,
            normalize_by_imagenet=True,
    ) -> None:
        # resampled patches... (for validation and test)
        self.normalize = normalize
        self.normalize_by_dataset = normalize_by_dataset
        self.normalize_by_imagenet = normalize_by_imagenet

        self.max_value = max_value

        self.saved_resampled_patches = saved_resampled_patches
        self.res_list = res_list
        self.allow_upsample = allow_upsample
        self.res_tolerance = res_tolerance

        self.auto_resample = auto_resample
        self.device = device
        self.edge_weight = edge_weight
        self.edge_threshold_pixel = edge_threshold_pixel
        self.edge_threshold_percent = edge_threshold_percent
        self.small_object_threshold = small_object_threshold
        self.dataset_split_by = dataset_split_by
        self.weighted_sampling = weighted_sampling
        self.epochs = epochs
        self.rescale = rescale
        self.append_data = append_data
        self.year_col = year_col
        self.target_features = target_features
        self.target_col = target_col
        self.rasterize_all_touched = rasterize_all_touched
        # For each feature, we create mask, get the count (maybe other characteristics in the future) so we need a identifier for raw features in their original format
        self.raw_feature_suffix = 'polygons'

        if target_classes is not None and 'bg' in target_classes:
            assert target_classes.index('bg') == 0
        self.target_classes = target_classes

        if target_features is not None:  # Case of prediction (No target is provided)
            if target_classes is None:
                if target_col is not None:
                    self.target_classes = pd.unique(target_features[self.target_col]).astype(str)
                else:
                    self.target_classes = ["1"]

        self.target_dtype = torch.long
        if self.target_classes is not None and len(self.target_classes) <= 2:
            self.target_dtype = torch.float32

        self.merge_mode = merge_mode
        self.allow_empty_patches = allow_empty_patches
        self.extract_images_for_areas = extract_images_for_areas
        self.extracted_image_col = "extracted_images"
        if self.target_features is not None:
            self.assigned_target_features = gpd.sjoin(
                self.target_features, labeled_areas, predicate="intersects", how="inner"
            ).rename(columns={"index_right": "area_id"})

            if self.year_col is not None:
                self.assigned_target_features[self.year_col] = pd.to_numeric(self.assigned_target_features[self.year_col])

            if self.target_col is not None:  # Case of multi-class segmentation
                self.assigned_target_features[f"{self.target_col}_"] = self.assigned_target_features[
                    self.target_col].map({cls_name: i for i, cls_name in enumerate(target_classes)})

                n_targets = len(self.assigned_target_features)
                self.assigned_target_features.dropna(subset=f"{self.target_col}_", axis=0, inplace=True)
                if n_targets > len(self.assigned_target_features):
                    logging.warning(f"{n_targets- len(self.assigned_target_features)} targets samples were dropped "
                                    f"since they did not correspond to the given mapping '(target_feature_names').")
        else:
            self.assigned_target_features = None

        # Only used here
        self.resample_uplim = 0.59
        self.scale_effi = 100

        self.weight_vars: typing.Union[typing.Literal["default", None], typing.List[str]] = ['patch_hectares',
                                                                                             'spatial_clusters',
                                                                                             'feature_counts',
                                                                                             'ori_resolution']
        # Cluster shapefile
        self.cluster_shp = {
            'file_path': '/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/WeightedSamping/countries.shp',
            'cluster_id_col': 'area_name', 'predicate': 'intersects'}
        # self.cluster_shp = {
        #     'file_path': '/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/WeightedSamping/clusters_30_bufferp0003_dissolved.shp',
        #     'cluster_id_col': 'CLUSTER__1', 'predicate': 'within'}

        self.biome_shp = None

        if "spatial_clusters" in self.weight_vars:
             assert self.cluster_shp is not None

        self.normalize = normalize
        # These values are based on RGBI images in denmark, spain, germany, and california...
        self.means = [0.407, 0.416, 0.377, 0.502]  # in the sequence of RGBI
        self.stds = [0.144, 0.123, 0.106, 0.140]

        self.normalize_by_dataset = normalize_by_dataset
        self.mean_map = {
            'germany20cm_2022': [0.4281, 0.482, 0.4505, 0.6711],  # in the sequence of RGBI
            'california60cm_2020': [0.4158, 0.4169, 0.3691, 0.498],
            'swiss25cm_2022': [0.4323, 0.5509, 0.5452, 0.525],
            'spain25cm_2022': [0.4861, 0.4531, 0.402, 0.],
            'finland40cm_2021': [0.3661, 0.3777, 0.3671, 0.3235],
            'denmark20cm_2020': [0.3543, 0.3946, 0.3799, 0.4767],
        }
        self.std_map = {
            'germany20cm_2022': [0.2006, 0.1738, 0.1549, 0.1651],
            'california60cm_2020': [0.1488, 0.1253, 0.1043, 0.1492],
            'swiss25cm_2022': [0.2235, 0.2173, 0.1597, 0.2564],
            'spain25cm_2022': [0.1605, 0.1402, 0.1373, 1e-6],
            'finland40cm_2021': [0.1684, 0.1604, 0.1474, 0.1396],
            'denmark20cm_2020': [0.1323, 0.1121, 0.0986, 0.1293],
        }
        self.mean_imagenet = [0.485, 0.456, 0.406, self.means[-1]]
        self.std_imagenet = [0.229, 0.224, 0.225, self.stds[-1]]


        super().__init__(
            labeled_areas=labeled_areas,
            images_df=images_df,
            reference_source=reference_source,
            project_crs=project_crs,
            target_features=target_features,  # e.g, tree polygons (hand labels).
            target_col=target_col,  # e.g 'feature_type', or 'tree_species'
            target_classes=target_classes,
            patch_size=patch_size,
            allow_partial_patches=allow_partial_patches,
            allow_empty_patches=allow_empty_patches,
            sequential_stride=sequential_stride,
            dataset_transform=dataset_transform,
            processed_dir=processed_dir,
            save_samples=save_samples,
            save_patch_df=save_patch_df,
            save_labeled_areas_df=save_labeled_areas_df,
            extract_images_for_areas=extract_images_for_areas,
            dataset_name=dataset_name,
            merge_mode=merge_mode,
            rasterize_all_touched=rasterize_all_touched,
        )

    @classmethod
    def init_from_config(cls, config: BaseSegmentationConfig, dataset_name: str, labeled_areas: gpd.GeoDataFrame,
                         images_df: gpd.GeoDataFrame, target_features: gpd.GeoDataFrame, dataset_transform=None,
                         **kwargs):
        """ Initialize the segmentation from a config file.
        """
        options = {
            "reference_source": config.get("reference_source"),
            "project_crs": config.get("project_crs"),
            "target_col": config.get("target_col"),
            "target_classes": config.get("target_classes"),
            "patch_size": config.get("patch_size"),
            "allow_partial_patches": config.get("allow_partial_patches"),
            "allow_empty_patches": config.get("allow_empty_patches"),
            "sequential_stride": config.get("sequential_stride"),
            "dataset_transform": dataset_transform,
            "processed_dir": config.get("processed_dir"),
            "save_samples": config.get("save_samples"),
            "save_patch_df": config.get("save_patch_df"),
            "save_labeled_areas_df": config.get("save_labeled_areas_df"),
            "extract_images_for_areas": config.get("extract_images_for_areas"),
            "dataset_name": dataset_name,
            "merge_mode": config.get("merge_mode"),
            "rasterize_all_touched": config.get("rasterize_all_touched"),
            "append_data": config.get("append_data"),
            "year_col": config.get("year_col"),
            "rescale": config.get("rescale"),
            "epochs": config.get("epochs"),
            "weighted_sampling": config.get("weighted_sampling"),
            "dataset_split_by": config.get("dataset_split_by"),
            "edge_weight": config.get('edge_weight'),
            "edge_threshold_pixel": config.get('edge_threshold_pixel'),
            "edge_threshold_percent": config.get('edge_threshold_percent'),
            "small_object_threshold": config.get('small_object_threshold'),
            "device": config.get('device'),
            "auto_resample": config.get('auto_resample'),
            "saved_resampled_patches": config.get('saved_resampled_patches'),
            "max_value": config.get("max_value"),
            "normalize": config.get("normalize"),
            "normalize_by_dataset": config.get("normalize_by_dataset"),
            "normalize_by_imagenet": config.get("normalize_by_imagenet"),
            # "res_list": config.get('res_list'),
            # "allow_upsample": config.get('allow_upsample'),
            # "res_torlerance": config.get('res_torlerance'),
            # "reverse_padding": config.get("reverse_padding"),
        }
        options.update(kwargs)
        return cls(labeled_areas=labeled_areas,
                   images_df=images_df,
                   target_features=target_features,
                   **options)


    def get_patch(self, idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        patch = self.patch_df.loc[idx]
        pt = patch.to_dict()
        pt.update(self.get_patch_raster_images(patch))

        image_srcs = self.images_df.src.unique()
        img_names = [i for i in image_srcs if i in pt.keys()]

        pt_ori_res = round(patch["ori_transform"][0], 2)
        assert img_names != []
        assert pt_ori_res > 0

        if not self.append_data:
            year_id = img_names[0].split('_')[-1] if self.year_col is not None else None

            if self.rescale:
                if round(pt_ori_res * 100, 0) - round(self.resample_uplim * 100, 0) < -2:
                    resample_lowlim = pt_ori_res + 0.02
                    scale_factors = torch.tensor(
                        [int(pt_ori_res / self.resample_uplim * self.scale_effi) / self.scale_effi,
                         int(pt_ori_res / resample_lowlim * self.scale_effi) / self.scale_effi],
                        dtype=torch.float32
                    )
                else:
                    scale_factors = torch.tensor([1, 1], dtype=torch.int8)
            else:
                scale_factors = None

            pt[self.reference_source] = pt.pop(img_names[0])

        pt['patch_id'] = idx

        if self.target_features is not None:
            tr_features = self.assigned_target_features.query(f"area_id == {pt['area_id']}")
            if self.year_col is not None:
                tr_features = tr_features.query(f"{self.year_col} == {int(year_id)}")
            tr_features = tr_features[tr_features.intersects(pt['geometry'])]
            tr_features_tr = tr_features.to_crs(patch['ori_crs'])

            if self.target_col is None:
                fr = tr_features_tr['geometry'].to_list()
            else:
                tr_features_tr = tr_features_tr.sort_values(f"{self.target_col}_")
                fr = list(tr_features_tr[['geometry', f"{self.target_col}_"]].itertuples(index=False, name=None))

            if len(fr) == 0:
                shape = patch["shape"]
                seg_msk = torch.zeros(shape, dtype=torch.float32)
                en = torch.zeros(shape, dtype=torch.float32)
                sx = torch.zeros(shape, dtype=torch.float32)
                sy = torch.zeros(shape, dtype=torch.float32)
            else:
                fr2 = ((geom, value + 1) for geom, value in zip(fr, list(range(len(fr)))))
                seg_msk = rasterio.features.rasterize(fr2, out_shape=patch["shape"], all_touched=self.rasterize_all_touched, fill=0, transform=patch["ori_transform"])
                seg_msk = torch.tensor(seg_msk.astype(np.float32), dtype=torch.float32)
                en, sx, sy = watershed_mask(fr, out_shape=patch["shape"], all_touched=self.rasterize_all_touched, fill=0, transform=patch["ori_transform"])

            weights = calculate_size_weights(seg_msk.long(), self.small_object_threshold, device=None)
            edge_weights = calculate_edge_weights(seg_msk.long(), torch.tensor(en, dtype=torch.float32), self.edge_weight, self.edge_threshold_pixel, self.edge_threshold_percent, device=None)

            gauss_density = gaussian_density(tr_features, sigma=1, out_shape=patch["shape"], transform=patch["ori_transform"])

            area = self.labeled_areas.loc[[patch.area_id]]
            area_tr = area.to_crs(patch['ori_crs'])
            mask = rasterio.features.rasterize(area_tr['geometry'].to_list(), out_shape=patch["shape"],all_touched=self.rasterize_all_touched, fill=0, transform=patch["ori_transform"])
            mask = torch.tensor(mask, dtype=torch.bool)

            img_data = pt[self.reference_source]
            img_mask = (img_data.sum(dim=0) == 0) | torch.isnan(img_data[0])

            mask = mask & ~img_mask

            self.reverse_padding = 10
            non_zeros = torch.nonzero(mask)
            row_start, col_start, row_end, col_end = non_zeros[:, 0].min(), non_zeros[:, 1].min(), non_zeros[:, 0].max(), non_zeros[:, 1].max()
            inner_mask = torch.zeros_like(mask, dtype=torch.bool)
            inner_mask[row_start + self.reverse_padding:row_end - self.reverse_padding, col_start + self.reverse_padding:col_end - self.reverse_padding] = 1
            ct_inner = len([i for i in torch.unique(seg_msk * (mask * inner_mask)) if i !=0]) if seg_msk.sum() > 0 else len(tr_features)
            ct = len([i for i in torch.unique(seg_msk * mask) if i != 0]) if seg_msk.sum() > 0 else len(tr_features)

            pt["target"] = seg_msk
            pt["target_mask"] = mask
            pt["energy"] = torch.tensor(en, dtype=torch.float32)
            pt["sobx"] = torch.tensor(sx, dtype=torch.float32)
            pt["soby"] = torch.tensor(sy, dtype=torch.float32)
            pt["density"] = torch.tensor(gauss_density, dtype=torch.float32)
            pt["count"] = torch.tensor(ct, dtype=torch.float32)
            pt["inner_mask"] = inner_mask
            pt['inner_count'] = torch.tensor(ct_inner, dtype=torch.float32)
            pt['count_weights'] = torch.tensor(weights, dtype=torch.float32)
            pt['edge_weights'] = torch.tensor(edge_weights, dtype=torch.float32)

            if scale_factors is not None:
                pt["scale_factors"] = torch.tensor((scale_factors * self.scale_effi), dtype=torch.int8)

            pt["ori_resolution"] = torch.round(torch.tensor(pt_ori_res, dtype=torch.float32) * 100) / 100
            nonmask_per = torch.sum(pt["target_mask"]) / (self.patch_size[0] ** 2)
            ha = (self.patch_size[0] * pt["ori_resolution"]) ** 2 * nonmask_per / 10000
            pt["hectares"] = torch.tensor(ha, dtype=torch.float32)

            if self.epochs == 1 and self.processed_dir is not None:
                data_out = tuple([pt[i] for i in image_srcs if i in pt.keys()]) + (
                    torch.tensor(np.expand_dims(en, axis=0)), torch.tensor(np.expand_dims(sx, axis=0)),
                    torch.tensor(np.expand_dims(sy, axis=0)),
                    seg_msk.unsqueeze(0), mask.unsqueeze(0), inner_mask.unsqueeze(0),
                    weights.unsqueeze(0), edge_weights.unsqueeze(0)
                )

                pt_2 = torch.vstack(data_out)

                base_dir = self.processed_dir / 'extracted_pts'
                base_dir.mkdir(exist_ok=True)
                cnt_new = f'{self.dataset_name}_{idx}'
                output_fp = base_dir / f"{cnt_new}.tif"
                ndt = None

                with rasterio.open(output_fp,
                                   mode='w',
                                   driver='GTiff',
                                   height=pt_2.shape[1],
                                   width=pt_2.shape[2],
                                   count=pt_2.shape[0],
                                   dtype='float32',
                                   crs=pt["ori_crs"],
                                   transform=pt["ori_transform"],
                                   tiled=False,
                                   interleave='pixel',
                                   nodata=ndt) as dst:
                    dst.write(pt_2.numpy())

        return pt

    def batch_to_raster(self, x, dataset_name='patch', out_dir='pts_raster', overwrite=False, crs=None, transform=None):
        patch_id, pt = x[:2]
        if len(x) > 2:
            crs, transform = x[2:]
        # patch_id = patch_id.cpu().numpy()
        # pt = pt.cpu().numpy()
        base_dir = Path(self.processed_dir) / out_dir
        base_dir.mkdir(exist_ok=True)
        cnt_new = '{}_{}'.format(dataset_name, patch_id)
        output_fp = base_dir / f"{cnt_new}.tif"
        ndt = None

        if not overwrite and os.path.exists(output_fp):
            return

        with rasterio.open(output_fp,
                           mode='w',
                           driver='GTiff',
                           height=pt.shape[2],
                           width=pt.shape[1],
                           count=pt.shape[0],
                           dtype='float32',
                           tiled=False,
                           interleave='pixel',
                           nodata=ndt,
                           crs=crs,
                           transform=transform,
                           ) as dst:
            dst.write(pt)

    def recalculate_weights(self, energy_layer, mask, device, calculate_cw=True, calculate_ew=False):
        """recalculate weights for the rescaled patches"""
        if calculate_cw:
            cw = calculate_size_weights(mask, small_object_threshold=self.small_object_threshold, device=device)
            return cw
        if calculate_ew:
            ew = calculate_edge_weights(mask, energy_layer, edge_weight=self.edge_weight, edge_threshold_pixel=self.edge_threshold_pixel, edge_threshold_percent=self.edge_threshold_percent, device=device)
            return ew

        else:
            return None

    def recalculate_en(self, idx, segmask, device=None):
        """recalculate energy band for the rescaled patches"""
        en, sx, sy = watershed_mask2(segmask[idx].squeeze(),self.patch_size)
        if device is not None:
            sobel = torch.stack([torch.from_numpy(sx), torch.from_numpy(sy)], dim=0).to(device)
            return sobel, torch.unsqueeze(torch.from_numpy(en).long().to(device), 0)
        else:
            sobel = torch.stack([torch.from_numpy(sx), torch.from_numpy(sy)], dim=0)
            return sobel, torch.unsqueeze(torch.from_numpy(en).long(), 0)

    def preprocess_patch(self, sample, max_value):
        x = torch.tensor(sample[self.reference_source].unsqueeze(0))
        channel_count = x.shape[1]
        x_m = ~torch.isnan(x)
        x = torch.where(x_m, x, torch.tensor(0.0))
        if max_value != 0:
            x = x / max_value

        assert channel_count <= 4, "The number of channels larger than number of normalization factors."

        # Normalization based on global stats
        if self.normalize:
            if self.normalize_by_dataset:
                assert len([i for i in sample['src_name'] if i not in self.mean_map.keys()]) == 0, f"Std not found for {sample['src_name']}"
                means = torch.tensor([self.mean_map.get(s, None) for s in sample['src_name']], dtype=torch.float32)
                stds = torch.tensor([self.std_map.get(s, None) for s in sample['src_name']], dtype=torch.float32)
                x = torch.reshape(torch.cat(tuple(map(lambda i: transforms.Normalize(means[i][:channel_count], stds[i][:channel_count])(x[i]), range(x.shape[0])))), x.shape)
            else:
                if self.normalize_by_imagenet:
                    means, stds = self.mean_imagenet, self.std_imagenet
                else:
                    means, stds = self.means, self.stds
                x = transforms.Normalize(means[:channel_count], stds[:channel_count])(x)

        return x

    def get_resampled_patch(self, idx: int) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Get resampled patches for each val and test set... would this work in training? resampled to the rest_list
        """

        # get original patch
        pt = self.get_patch(idx)

        # start to resample
        x = self.preprocess_patch(pt, self.max_value)
        # print(x.sum())
        ys = [torch.tensor(pt["target"]).unsqueeze(0).unsqueeze(0),
              torch.tensor(torch.stack([pt["sobx"], pt["soby"]])).unsqueeze(0),
              torch.tensor(pt["energy"]).unsqueeze(0).unsqueeze(0),
              torch.tensor(pt["density"]).unsqueeze(0).unsqueeze(0),
              torch.tensor(pt["inner_mask"]).unsqueeze(0).unsqueeze(0),
              torch.tensor(pt['count_weights']).unsqueeze(0).unsqueeze(0),
              torch.tensor(pt['edge_weights']).unsqueeze(0).unsqueeze(0),
              torch.tensor(pt["inner_count"]).unsqueeze(0),
              torch.tensor(pt['patch_id']).unsqueeze(0),
              torch.tensor(pt['hectares']).unsqueeze(0),
              torch.tensor(pt['ori_resolution']).unsqueeze(0)]

        ha, ori_res, patch_ids, ycount, count_weights, edge_weights = torch.tensor([ys[-2]]), torch.tensor([ys[-1]]), torch.tensor([ys[-3]]), torch.tensor([ys[-4]]),torch.tensor(ys[-6]), torch.tensor(ys[-5])
        ys = [y.clone().detach() for y in ys[:5]]
        ydtype = [y.dtype for y in ys]
        # Attach resampled ones to original evaluation batch (20, 30, 40, 50, 60)
        ori_res = torch.round(ori_res * 100) / 100
        unique_res = torch.unique(torch.round(ori_res * 100) / 100)

        if self.allow_upsample:
            process_res = unique_res
        else:
            process_res = torch.tensor([i for i in unique_res if abs(i - 0.6) < self.res_tolerance])

        if len(process_res) == 0:
            ys.append(torch.tensor(count_weights, dtype=torch.float32))
            ys.append(torch.tensor(edge_weights, dtype=torch.float32))
            ys.append(torch.tensor(ycount, dtype=torch.float32))
            ys.append(torch.tensor(ha, dtype=torch.float32))
            ys.append(torch.tensor(ori_res, dtype=torch.float32))
            ys.append(torch.tensor(patch_ids, dtype=torch.float32))
            ys.append(torch.tensor([1.] * len(patch_ids), dtype=torch.float32))
            return x, ys

        # Identify the patches that need resampling
        idxs = [torch.tensor([i for i, tupl in enumerate(ori_res) if tupl == j]) for j in process_res]
        has_ = [torch.tensor(ha[j]) for j in idxs]
        oris_ = [torch.tensor(ori_res[j]) for j in idxs]
        ycounts_ = [torch.tensor(ycount[j]) for j in idxs]

        to_res = [[i for i in self.res_list if i > j] for j in process_res] if not self.allow_upsample else [[i for i in self.res_list if abs(i - j) > self.res_tolerance] for j in process_res]
        scales_ = [torch.round(torch.tensor([j / t if abs(j - t) > self.res_tolerance else 1. for t in i]) * 100) / 100 for i, j in zip(to_res, process_res)]
        unique_scales = torch.unique(torch.cat(scales_))
        ys = [y.to(dtype=torch.float32) for y in ys]

        # resampling
        def f(m):
            pi_list, scl = m
            ys_p = [ite[pi_list] for ite in ys]
            aug_func = AugmentationSequential(
                RandomAffine(degrees=0.0, scale=torch.tensor([scl, scl]), p=1.0),
                same_on_batch=True,
                data_keys=['input', *['mask'] * len(ys_p)],
            )
            out = aug_func(x[pi_list], *ys_p)
            return out

        params_all = [(torch.tensor([a, h, r, c]), b) for j in range(len(idxs)) for (a, h, r, c), b in itertools.product(zip(idxs[j], has_[j], oris_[j], ycounts_[j]), scales_[j])]
        params = [[i[0].item() for i, j in params_all if j == j0] for j0 in unique_scales]
        out_list = list(map(f, list(zip(params, unique_scales))))
        x_out = torch.cat([i[0] for i in out_list])
        y_out = [torch.cat([y[j] for y in [i[1:] for i in out_list]]) for j in range(len(ys))]

        # Recalculate count for the upper sampled ones
        ycount2 = torch.tensor([len([j for j in torch.unique(y_out[0][i] * y_out[4][i]) if j != 0]) for i in range(y_out[0].shape[0])])
        idx_scale = [j for i in zip(params, unique_scales.numpy()) for j in itertools.product(i[0], [i[1]])]
        # get params_idx based on idx_scale
        params_all_reorder_idx = [idx for k in idx_scale for idx, i in enumerate(params_all) if i[0][0] == k[0] and i[-1] == k[1]]
        # replace the second values in params_all with ycount2
        for (params_idx, yc2) in zip(params_all_reorder_idx, ycount2):
            params_all[params_idx][0][3] = yc2

        # Recalculate energy bands for the resampled patches
        rs = list(map(functools.partial(self.recalculate_en, segmask=y_out[0]), range(y_out[0].shape[0])))
        # replace augmented en, sx, sy to the original ones
        y_out[1] = torch.stack([i[0] for i in rs]).to(torch.float32)
        y_out[2] = torch.stack([i[1] for i in rs]).to(torch.float32)
        y_out = [y.to(dtype=yt, device='cpu') for y, yt in zip(y_out, ydtype)]
        del rs

        # Recalculate weights for the resampled patches
        weights_new = self.recalculate_weights(energy_layer=y_out[2], mask=y_out[0], calculate_cw=True, calculate_ew=False, device=None)
        count_weights_rescale = weights_new.to(torch.float32)
        del weights_new

        weights_new = self.recalculate_weights(energy_layer=y_out[2], mask=y_out[0], calculate_cw=False, calculate_ew=True, device=None)
        edge_weights_rescale = weights_new.to(torch.float32)
        del weights_new

        # other information
        has_all = torch.cat([torch.tensor([i[1].item() for i, j in params_all if j == j0]) for j0 in unique_scales])
        oris_all = torch.cat([torch.tensor([i[2].item() for i, j in params_all if j == j0]) for j0 in unique_scales])
        ycounts_all = torch.cat([torch.tensor([i[3].item() for i, j in params_all if j == j0]) for j0 in unique_scales])
        scales_all = torch.cat([torch.tensor([j.item() for i, j in params_all if j == j0]) for j0 in unique_scales])

        ids_all = torch.cat([torch.tensor([i[0].item() for i, j in params_all if j == j0]) for j0 in unique_scales]).to(torch.int8)
        new_res = torch.tensor([b / a for a, b in zip(scales_all, oris_all)], dtype=torch.float32)
        patch_ids_all = torch.tensor([patch_ids[a.item()].item() for i, a in enumerate(ids_all.to(torch.int8))])

        # append the unprocessed patches
        x_out = torch.cat([x_out, x]).to(torch.float32)
        y_out = [torch.cat([y_out[j], ys[j]]) for j in range(len(ys))]
        y_out = [y.to(dtype=yt) for y, yt in zip(y_out, ydtype)]

        count_weights_all = torch.cat([count_weights_rescale, count_weights])
        edge_weights_all = torch.cat([edge_weights_rescale, edge_weights])
        ycounts_all = torch.cat([ycounts_all, ycount])
        has_all = torch.cat([has_all, ha])
        patch_res_all = torch.cat([new_res, ori_res])
        patch_ids_all = torch.cat([patch_ids_all, patch_ids])
        scales_all = torch.cat([scales_all, torch.tensor([1.] * x.shape[0])])

        y_out.append(torch.tensor(count_weights_all, dtype=torch.float32))
        y_out.append(torch.tensor(edge_weights_all, dtype=torch.float32))
        y_out.append(torch.tensor(ycounts_all, dtype=torch.float32))
        y_out.append(torch.tensor(has_all, dtype=torch.float32))
        y_out.append(torch.tensor(patch_res_all, dtype=torch.float32))
        y_out.append(torch.tensor(patch_ids_all, dtype=torch.float32))
        y_out.append(torch.tensor(scales_all, dtype=torch.float32))

        pt[self.reference_source] = x_out
        pt["target"] = y_out[0]
        pt["sobx"] = y_out[1][:, 0, :, :].unsqueeze(1)
        pt["soby"] = y_out[1][:, 1, :, :].unsqueeze(1)
        pt["energy"] = y_out[2]
        pt["density"] = y_out[3]
        pt["inner_mask"] = y_out[4]
        pt['count_weights'] = y_out[5]
        pt['edge_weights'] = y_out[6]
        pt["inner_count"] = y_out[7]
        pt['hectares'] = y_out[8]
        pt['ori_resolution'] = y_out[9]
        pt['patch_id'] = y_out[10]
        pt['scale'] = y_out[11]

        # save resampled patches
        if self.epochs <= 1 and self.processed_dir is not None:
            fn_names = ['_'.join([str(pi.item()), str(int(res.item() * 100))]) for pi, res in zip(patch_ids_all, patch_res_all)]
            batch_out = []
            batch_out.append(x_out)
            batch_out.extend([y_out[i] for i in range(len(y_out[:7]))])
            batch_out = torch.cat(batch_out, 1)
            list(map(functools.partial(self.batch_to_raster, dataset_name=self.dataset_name, out_dir='auto_resampled_pts', overwrite=True), list(zip(fn_names, batch_out.cpu()))))

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

        image_keys = np.unique(assigned_images.src.tolist())
        if len(image_keys) > 1 and self.append_data is False:
            complete_areas = []
            for ik in image_keys:
                complete_areas.extend(assigned_images.query("src == @ik").area_id.tolist())
        else:
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

    def build_patch_table_sequentially(self, patch_start_index=0):

        # labeled area can be larger than masked layer...
        # labeled area can be used to extract images, and then mask based on the intersection of
        # the patch grids and masked layer and image bounds...

        patches = []
        height, width = self.patch_size

        image_keys = np.unique(list(self.assigned_images.src))
        if len(image_keys) > 1 and self.append_data is False:
            images_grouped_by_area = self.assigned_images[self.assigned_images.src.isin(image_keys)].groupby('area_id')
            year_id = image_keys[0].split('_')[-1] if self.year_col is not None else None
        else:
            images_grouped_by_area = self.assigned_images.query(f"src == '{self.reference_source}'").groupby('area_id')

        if self.weighted_sampling:
            weight_vars = []
            if hasattr(self, "cluster_shp") and self.cluster_shp is not None and "spatial_clusters" in self.weight_vars:
                cluster_id_col = self.cluster_shp.get('cluster_id_col')
                cluster_df = gpd.read_file(self.cluster_shp.get('file_path'))
                predicate = self.cluster_shp.get('predicate')

                if 'uid' in cluster_df.columns:
                    cluster_df.drop('uid', inplace=True, axis=1)
                cluster_df = cluster_df.reset_index().rename(columns={'index': 'uid'})
                # if 'uid' in self.labeled_areas.columns:
                #     self.labeled_areas.drop('uid', inplace=True, axis=1)
                # self.labeled_areas = self.labeled_areas.reset_index().rename(columns={'index': 'uid'})

                if 'uid' not in self.labeled_areas.columns:
                    self.labeled_areas = self.labeled_areas.reset_index().rename(columns={'index': 'uid'})
                cluster_df = cluster_df.to_crs(self.project_crs)
                join_ = gpd.sjoin(self.labeled_areas, cluster_df, how='left', predicate=predicate)
                if join_[cluster_id_col].isnull().sum() > 0:
                    logging.warning(f"{join_[cluster_id_col].isnull().sum()} patches do not intersect "
                                    f"with the geometry of spatial clusters. The cluster id for these "
                                    f"patches are assinged as others.")
                    join_[cluster_id_col] = join_[cluster_id_col].fillna('others')
                weight_vars.append('spatial_clusters')
            if self.target_features is not None and "feature_counts" in self.weight_vars:
                weight_vars.append("feature_counts")
            [weight_vars.append(i) for i in self.weight_vars if not i in(["feature_counts", "spatial_clusters"])]
            assert weight_vars is not []

        for area_idx, area_images in tqdm(images_grouped_by_area, desc="Iterating over areas"):
            area = self.labeled_areas.loc[area_idx]  # Returns a series
            img_indices = area_images.reset_index()["image_id"].values

            imgs = self.images_df.loc[img_indices]

            ori_crs, ori_transform, union_geometry, nodatavals, src_name = self.get_imgs_reference(imgs)
            # We still want to work with the overlapping geometry instead of the pure image geometry
            # union_geometry = union_geometry.intersection(area.geometry)

            # Do not use overlapping geometry but discard patches do not intersect with image bounds in the end of
            # this function...
            union_geometry_img = copy(union_geometry)
            union_geometry = copy(area.geometry)

            # Now we divide this geometry into smaller patches. Use the point approach; start with the top left point,
            # get positions in image space
            col_start, col_end, row_start, row_end = self.get_rows_cols_from_geom(
                ori_crs, ori_transform, union_geometry
            )

            if row_start > row_end or col_start > col_end:
                logging.info(f"row_start > row_end or col_start > col_end {row_start} {row_end} {col_start} {col_end}")

            stride_row, stride_col = self.sequential_stride
            if self.allow_partial_patches:
                col_itr_start = col_start
                row_itr_start = row_start
                col_itr_end = col_end
                row_itr_end = row_end
            else:
                # use center
                col_itr_start = col_start + ((col_end - col_start) % height) // 2
                row_itr_start = row_start + ((row_end - row_start) % width) // 2
                col_itr_end = col_end - height
                row_itr_end = row_end - width

            for col, row in product(
                    range(col_itr_start, col_itr_end, stride_col), range(row_itr_start, row_itr_end, stride_row)
            ):
                # Window is defined as offset in pure image space
                wn = windows.Window(col, row, width, height)
                wx, wy = rasterio.transform.xy(ori_transform, [row, row + width], [col, col + height])
                # Window box in original CRS
                wn_geometry = box(wx[0], wy[1], wx[1], wy[0])  # left, bottom, right, top
                transform_in_ori_crs = windows.transform(wn, ori_transform)

                patch = {
                    "ori_geometry": wn_geometry,
                    "geometry": wn_geometry,  # !!This is a placeholder which is transformed to project_crs in the end
                    "shape": (height, width),
                    "wn_ori": (col, row, width, height),
                    "ori_crs": ori_crs,
                    "ori_transform": transform_in_ori_crs,
                    "area_id": area_idx,
                    "nodatavals": nodatavals,  # nodata values for images overlapping with the patch geometry
                    "src_name": src_name,  # the name of image sources, which could taking the variations in sensors
                    # into consideration and reflectance difference depending on the ground obj..
                }

                # variables used to implement stratified sampling / weighted sampling
                if self.weighted_sampling:
                    patch.update({"weight_vars": weight_vars})
                    if "patch_hectares" in weight_vars:
                        ha = width * height * transform_in_ori_crs[0] * (transform_in_ori_crs[4] * -1) / 10000
                        patch.update({"patch_hectares": ha})
                    if "feature_counts" in weight_vars:
                        # It may be faster to do this way instead of the direct intersection.
                        tr_features = self.assigned_target_features.query(f"area_id == {area_idx}")  # Target features
                        tr_features = tr_features.query(
                            f"{self.year_col} == {int(year_id)}") if self.year_col is not None else tr_features
                        tr_features = tr_features[
                            tr_features.intersects(geometry_to_crs(wn_geometry, ori_crs, self.project_crs))]
                        patch.update({"feature_counts": len(tr_features)})
                    if "spatial_clusters" in weight_vars:
                        cluster_id = join_[join_['uid_left'] == area.uid][cluster_id_col].tolist()[0]
                        patch.update({"spatial_clusters": cluster_id})

                union_geometry_img = geometry_to_crs(union_geometry_img, self.project_crs, ori_crs)

                if self.dataset_split_by == 'areas' and (wn_geometry.intersects(union_geometry_img) and wn_geometry.intersection(union_geometry_img).area
                        / wn_geometry.area > 0.3):
                    patches.append(patch)
                else:
                    patches.append(patch)
                # wn_union = rasterio.windows.from_bounds(
                #     *geometry_to_crs(union_geometry, self.project_crs, ori_crs).bounds, ori_transform)
                # wn_union = rasterio.windows.Window(np.floor(wn_union.col_off), np.floor(wn_union.row_off),
                #                                  round(wn_union.width),
                #                                  round(wn_union.height))
                # try:
                #     windows.intersection(wn, wn_union)
                #     patches.append(patch)
                # except:
                #     continue

        pdf = gpd.GeoDataFrame.from_dict(patches).set_crs(self.project_crs)
        pdf = get_df_in_single_crs(pdf, self.project_crs)

        def features_count(rw, atf=self.assigned_target_features):
            return len(get_patch_features(atf, rw.geometry, area_id=rw.area_id))

        if not (self.allow_empty_patches or self.target_features is None):  # 2. Case of prediction
            pdf['n_target_features'] = pdf.apply(features_count, axis=1)
            pdf = pdf.query("n_target_features > 0")

        pdf['patch_id'] = range(patch_start_index, len(pdf) + patch_start_index)

        # verify that patches overlap with labeled_area they come from and adhere to their shape
        pdfs = []
        for area_idx in self.labeled_areas.index:
            pdfs.append(
                pdf.query("area_id == @area_idx").sjoin(
                    self.labeled_areas.query("index == @area_idx"), how="inner", predicate="intersects", lsuffix=""
                )[pdf.columns]
            )
        pdf = pd.concat(pdfs)
        pdf.drop_duplicates("patch_id", inplace=True)

        pdf['patch_id'] = range(patch_start_index, len(pdf) + patch_start_index)

        pdf.set_index("patch_id", inplace=True)

        if self.processed_dir is not None:
            p = self.processed_dir / 'qgis' / self.dataset_name
            logging.info(f"Writing labeled areas, image df and assigned images to {p}")
            p.mkdir(exist_ok=True, parents=True)
            pdf[['geometry', 'area_id']].to_file(p / "patch_grid.gpkg", driver="GPKG")
            self.labeled_areas[['geometry']].to_file(p / "labeled_areas.gpkg", driver="GPKG")
            self.images_df[['geometry', 'path']].to_file(p / "image_df.gpkg", driver="GPKG")
            self.assigned_images[['geometry']].to_file(p / "assigned_images.gpkg", driver="GPKG")
            if self.assigned_target_features is not None:
                self.assigned_target_features[['geometry']].to_file(p / "assigned_target_features.gpkg", driver="GPKG")

        return pdf, len(pdf)

    def get_imgs_reference(self, imgs):
        # if more than one image from the reference source is available for the patch, then create a virtual raster
        if len(imgs) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                idx_largest_overlap = np.argmax(imgs.area)
            union_geometry = unary_union(imgs.geometry.to_list())
            ori_transform = imgs["ori_transform"].iloc[idx_largest_overlap]
            ori_crs = imgs["ori_crs"].iloc[idx_largest_overlap]
            nodata = imgs['nodatavals'].iloc[idx_largest_overlap]
            src_name = imgs["src"].iloc[idx_largest_overlap]
        else:
            union_geometry = imgs["geometry"].iloc[0]
            ori_transform = imgs["ori_transform"].iloc[0]
            ori_crs = imgs["ori_crs"].iloc[0]
            nodata = imgs['nodatavals'].iloc[0]
            src_name = imgs["src"].iloc[0]
        return ori_crs, ori_transform, union_geometry, nodata, src_name

