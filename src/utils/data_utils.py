import logging
import os
import shutil
import time

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

from src.data.base_segmentation_dataset import assign_target_polygons_to_areas


def is_same_crs(crs1, crs2):
    if hasattr(crs1, 'crs'):
        crs1 = getattr(crs1, 'crs')
    if hasattr(crs2, 'crs'):
        crs2 = getattr(crs2, 'crs')

    c1 = pyproj.crs.CRS(crs1)
    return c1.is_exact_same(crs2)


def get_common_bounds(g1, g2):
    return box(*g1).intersection(box(*g2)).bounds


def get_complete_areas(images_df, labeled_areas, reference_source):
    # create a lookup table (already taking the needed infos (needs more ram but is ok))
    assigned_images = gpd.sjoin(
        images_df[["src", "geometry"]], labeled_areas[["geometry"]],
        predicate="intersects", how="inner"
    ).rename(columns={"index_right": "area_id"})

    # filter out not found values
    missing_areas = labeled_areas.query("index not in @assigned_images.area_id")
    labeled_areas = labeled_areas.query("index in @assigned_images.area_id")
    if len(missing_areas) > 0:
        logging.info(f"{len(missing_areas)} areas with no image source, idx: {missing_areas}")
        logging.info(f"ignoring them for now")

    complete_areas = assigned_images.query("src == @reference_source").area_id
    return list(complete_areas)


def stratified_random_split(l_areas, images_df, config, target_features=None, rnds=11, write_split=False):
    areas = l_areas.copy()
    # assert hasattr(config, 'split_col') and config.split_col is not None
    split_col = config.split_col

    if split_col in areas.columns:
        logging.info("Reading existing dataset split from areas dataframe")
        train_areas = areas.query(f"{split_col} == 'train'")
        val_areas = areas.query(f"{split_col}  == 'val'")
        test_areas = areas.query(f"{split_col} == 'test'")
    # Case of a binary segmentation; Assign areas randomly
    elif config.target_col is None or (config.target_col is not None and target_features is None):
        complete_areas = get_complete_areas(images_df, areas, config.reference_source)
        labeled_areas = areas.query("index in @complete_areas")

        indices = labeled_areas.index.tolist()
        rs = np.random.RandomState(rnds)
        rs.shuffle(indices)
        l = len(indices)
        val_ratio = config.val_split
        test_ratio = config.test_split
        va = max(int(val_ratio * l), 1)
        ta = max(int((val_ratio + test_ratio) * l), va + 1)
        val_indices = indices[: va]  # First n to val
        test_indices = indices[va: ta]  # Middle n to test
        train_indices = indices[ta:]  # Rest to train

        areas[split_col] = 'undefined'
        areas.loc[train_indices, split_col] = 'train'
        areas.loc[val_indices, split_col] = 'val'
        areas.loc[test_indices, split_col] = 'test'
        if write_split:
            write_area_split(areas, config)

        train_areas = areas.loc[train_indices]
        val_areas = areas.loc[val_indices]
        test_areas = areas.loc[test_indices]
    else:
        if target_features is None:
            raise NotImplementedError("Creating of new splits for multi-class image classification not supported.")

        logging.info(f"Creating new dataset splits and write_split {write_split}")
        # feature_col = config.target_col
        # feature_classes = config.target_classes
        val_ratio = config.val_split
        test_ratio = config.test_split

        rs = np.random.RandomState(rnds)
        t0 = time.time()
        complete_areas = get_complete_areas(images_df, areas, config.reference_source)
        labeled_areas = areas.query("index in @complete_areas")
        t1 = time.time()

        assigned_target_features = assign_target_polygons_to_areas(
            labeled_areas, target_features)
        t2 = time.time()
        logging.info(f"complete labels:{t1 - t0},  Time for area-polygon join : {t2 - t1}")

        assert not assigned_target_features.empty

        # Split the smallest class into 3 parts (e.g. 10, 10, rest), and so on until no area is left
        train_indices = []
        val_indices = []
        test_indices = []
        covered_indices = []

        class_count = {}
        for k in config.target_classes:
            if k not in assigned_target_features.query(f"{config.target_col} == @k").values:
                class_count[k] = 0
            else:
                tfc = assigned_target_features.query(f"{config.target_col} == @k")
                class_count[k] = len(tfc)

        class_count = dict(sorted(class_count.items(), key=lambda item: item[1]))

        # Iterate in order of class with least occurences to most occurences and split into val, test and train
        for k in class_count:
            indices = assigned_target_features.query(f"{config.target_col} == @k").area_id.tolist()
            indices = set(indices)
            indices = [l for l in indices if
                       l not in covered_indices]  # Ignore the area that have been already assigned.

            l = len(indices)
            rs.shuffle(indices)
            val_indices.extend(indices[: int(val_ratio * l)])  # First n to val
            test_indices.extend(indices[int(val_ratio * l): int((val_ratio + test_ratio) * l)])  # Middle n to test
            train_indices.extend(indices[int((val_ratio + test_ratio) * l):])  # Rest to train
            covered_indices.extend(indices)
        indices = labeled_areas.loc[~labeled_areas.index.isin(covered_indices)].index.tolist()
        l = len(indices)
        val_indices.extend(indices[: int(val_ratio * l)])  # First n to val
        test_indices.extend(indices[int(val_ratio * l): int((val_ratio + test_ratio) * l)])  # Middle n to test
        train_indices.extend(indices[int((val_ratio + test_ratio) * l):])  # Rest to train
        covered_indices.extend(indices)

        if len(val_indices) == 0:
            logging.info("Split produced no val area, assigning from train")
            val_indices = [train_indices.pop(np.random.randrange(len(train_indices)))]
        if len(test_indices) == 0:
            logging.info("Split produced no test area, assigning from train")
            test_indices = [train_indices.pop(np.random.randrange(len(train_indices)))]

        logging.info(f"Split counts; val: {len(val_indices)}, test: {len(test_indices)}, train: {len(train_indices)}")
        assert len(val_indices) + len(test_indices) + len(train_indices) == len(set(complete_areas))

        areas[
            split_col] = 'undefined'  # It may be that certain areas can't be used. E.g., if no overlapping image is found.
        areas.loc[train_indices, split_col] = 'train'
        areas.loc[val_indices, split_col] = 'val'
        areas.loc[test_indices, split_col] = 'test'
        if write_split:
            write_area_split(areas, config)

        train_areas = areas.loc[train_indices]
        val_areas = areas.loc[val_indices]
        test_areas = areas.loc[test_indices]

    return train_areas, val_areas, test_areas


# Source: https://gitlab.com/rscph/planetmosaic/-/blob/main/core/histmatch_mosaic.py
# Author: Florian Reiner
def histogram_match(input_img, ref_img, nodata_val=0, plot_filename=None, **kwargs):
    """
    Match the input image to the reference image using histogram equalisation.
    No-data values are ignored during the histogram and cumulative distribution function calculation.
    """

    # Initialise result with zero, which is our no-data value
    out_img = np.zeros(input_img.shape)

    # Match histograms independently per band
    for band in range(input_img.shape[0]):

        # Convert source and ref to numpy masked arrays to handle no-data values correctly
        input_band = np.ma.masked_array(input_img[band, ...], mask=(input_img[band, ...] == nodata_val))
        ref_img[np.isnan(ref_img)] = nodata_val
        ref_band = np.ma.masked_array(ref_img[band, ...], mask=(ref_img[band, ...] == nodata_val))

        # Get unique pixel values, their counts and indices; unique values are sorted in ascending order
        input_values, input_idxs, input_counts = np.unique(input_band.ravel(), return_counts=True, return_inverse=True)
        ref_values, ref_counts = np.unique(ref_band.ravel(), return_counts=True)

        # Remove counts/values of no-data pixels
        input_counts = input_counts[~input_values.mask]
        ref_counts = ref_counts[~ref_values.mask]
        ref_values = ref_values[~ref_values.mask]

        if len(ref_counts) == 0:
            raise ValueError("Reference basemap has only nodata for this scene")

        # Get cumulative distribution functions for input and ref images, normalised to total number of pixels
        input_cdf = np.cumsum(input_counts)
        input_cdf = input_cdf / input_cdf[-1]
        ref_cdf = np.cumsum(ref_counts)
        ref_cdf = ref_cdf / ref_cdf[-1]
        if plot_filename:
            plt.figure()
            plt.plot(input_values[:len(input_cdf)], input_cdf, label="Input cdf")
            plt.plot(ref_values[:len(ref_cdf)], ref_cdf, label="Reference cdf")
            plt.legend()
            plt.title(plot_filename)
            plt.savefig(plot_filename)

        # Linearly interpolate ref values between input and ref cumulative distribution functions
        if 'filter' in kwargs:
            filter = kwargs['filter']

            if filter == 'linspace_cutoff':
                cutoff = kwargs['cutoff'] if 'cutoff' in kwargs else 0.02
                assert cutoff > 0 and cutoff < 1, "Cutoff must be between 0 and 1"
                # How many values to replace with linspace
                li = int(len(ref_values) * cutoff)
                assert li > 0 and li < len(ref_values), "Cutoff too high, no values left for interpolation"

                lvs = np.arange(1, li + 1) * (ref_values[li] - ref_values[li + 1]) + ref_values[li]
                rvs = np.arange(1, li + 1) * (ref_values[-li - 1] - ref_values[-li - 2]) + ref_values[-li]
                ref_values[:li] = lvs[::-1]
                ref_values[-li:] = rvs

            if filter == 'gaussian':
                sigma = kwargs['sigma'] if 'sigma' in kwargs else 0.5
                ref_values = gaussian_filter(ref_values, sigma=sigma)

        interpolated_vals = np.interp(input_cdf, ref_cdf, ref_values)
        interpolated_vals = np.append(interpolated_vals, nodata_val)  # no-data values are in the last bin of input_idxs

        # Map interpolated values to pixels and reshape back to band shape
        out_img[band, ...] = interpolated_vals[input_idxs].reshape(input_band.shape)

    return out_img


def write_area_split(areas, config):
    area_file = os.path.join(config.data_dir, config.labeled_areas_file)
    backup_area_file = os.path.join(config.data_dir, 'ori_' + config.labeled_areas_file)
    try:
        if area_file.endswith('shp'):
            raise ("Can't handle shapefiles")
        logging.info(f"Original areas are written to {backup_area_file}")
        shutil.move(area_file, backup_area_file)
        drv = 'GPKG'
        areas.to_file(area_file, driver=drv)
        return True
    except Exception as e:
        logging.error(f"Failed to write area split: {e}")
        try:
            shutil.move(backup_area_file, area_file)
        except Exception as e:
            logging.error(f"{e}")
        return False


def write_geotiff_to_file(count, height, width, bounds,
                          file_path='./unit_test/sample_labels/classification_img_sample.tif', crs='EPSG:4326',
                          dtype=rasterio.float32, data='random'):
    if not os.path.isfile(file_path):
        # Same as west, south, east, north
        minx, miny, maxx, maxy = bounds
        transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, width, height)
        if type(data) == str and data == 'random':
            rd = np.random.random((count, height, width)).astype(dtype=dtype)
        elif type(data) == np.ndarray:
            assert data.shape == (count, height, width), 'data must have shape (channels, height, width)'
            rd = data
        else:
            raise ValueError('data must be "random" or a numpy array')
        with rasterio.open(file_path,
                           mode='w',
                           driver='GTiff',
                           height=height,
                           width=width,
                           count=count,
                           dtype=dtype,
                           crs=crs,
                           transform=transform,
                           compress='lzw'
                           ) as dst:
            dst.write(rd)
