import logging
from itertools import product
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import torch
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm

from config.base import BaseSegmentationConfig
from src.data.base_dataset import BaseDataset, geometry_to_crs, get_df_in_single_crs
from src.data.image_extractor import extract_images


def get_common_bounds(g1, g2):
    return box(*g1).intersection(box(*g2)).bounds


def assign_target_polygons_to_areas(
        labeled_areas, target_features):
    # Assign each polygon/targeted feature to its respective labelled area
    assigned_target_features = gpd.sjoin(
        target_features, labeled_areas, predicate="intersects", how="inner"
    ).rename(columns={"index_right": "area_id"})
    return assigned_target_features


def calculate_boundary_weights(polygons: gpd.GeoDataFrame, scale: float):
    """Find boundaries between close polygons.

    Scales up each polygon, then get overlaps by intersecting. The overlaps of the scaled polygons are the boundaries.
    Returns geopandas data frame with boundary polygons.
    """
    # Scale up all polygons around their center, until they start overlapping
    # NOTE: scale factor should be matched to resolution and type of forest
    scaled_polys = gpd.GeoDataFrame({"geometry": polygons.geometry.scale(xfact=scale, yfact=scale, origin='center')})

    # Get intersections of scaled polygons, which are the boundaries.
    boundaries = []
    for i in range(len(scaled_polys)):

        # For each scaled polygon, get all nearby scaled polygons that intersect with it
        nearby_polys = scaled_polys[scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)]

        # Add intersections of scaled polygon with nearby polygons [except the intersection with itself!]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                boundaries.append(scaled_polys.iloc[i].geometry.intersection(nearby_polys.iloc[j].geometry))

    # Convert to df and ensure we only return Polygons (sometimes it can be a Point, which breaks things)
    boundaries = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(boundaries)}).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # If we have boundaries, difference overlay them with original polygons to ensure boundaries don't cover labels
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how='difference')
    if len(boundaries) == 0:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries


class BaseSegmentationDataset(BaseDataset):
    """
    BaseSegmentationDataset is designed to handle multiple sources of images at once.
    These sources may have different resolutions, channels or CRS.
    And the dataset would return a sample image accordingly. However, these flexibility has some limitations.
    For example, the expected size in different resolution maybe a couple of bit off,
    (assuming that the perfectly overlap).
    More likely, a sample may lie across multiple tiles in some sources,
    for that matter mapping each point on earth is still a very challenging task and it has its limitations.
    While we find work around these limitations, it may not be possible always.

    Note, that all the shape of patch, aoi and stride are given specific to a input image source (assuming there are multiple),
    and rest are of the shapes are sizes are calculated proportionally.
    """

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
            rasterize_all_touched: bool = False
    ) -> None:
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

        super().__init__(
            labeled_areas=labeled_areas,
            images_df=images_df,
            reference_source=reference_source,
            patch_size=patch_size,
            sequential_stride=sequential_stride,
            allow_partial_patches=allow_partial_patches,
            dataset_transform=dataset_transform,
            processed_dir=processed_dir,
            save_samples=save_samples,
            save_patch_df=save_patch_df,
            save_labeled_areas_df=save_labeled_areas_df,
            project_crs=project_crs,
            dataset_name=dataset_name
        )

    def filter_target_features(self):
        if self.target_col is None:
            return self.target_features
        else:
            nfl = self.target_features[
                ~self.target_features[self.target_col].isin(self.target_classes)]
            logging.info(
                "Following classes are not provided in target_classes, removing their features/polygons.")
            logging.info(f"Removing: {nfl[self.target_col].unique()}")

            # Notice the lack of ~ in the beginning.
            return self.target_features[
                self.target_features[self.target_col].isin(self.target_classes)]

    def get_patch(self, idx):
        # Returns a dict per source
        patch = self.patch_df.loc[idx]
        pt = patch.to_dict()
        pt.update(self.get_patch_raster_images(patch))
        pt['patch_id'] = idx

        if self.target_features is not None:
            # It may be faster to do this way instead of the direct intersection.
            tr_features = self.assigned_target_features.query(f"area_id == {pt['area_id']}")  # Target features
            tr_features = tr_features[tr_features.intersects(pt['geometry'])]
            tr_features_tr = tr_features.to_crs(patch['ori_crs'])
            if self.target_col is None:  # Binary segmentation; all features belong to the same class
                fr = tr_features_tr['geometry'].to_list()
            else:  # Multi-class segmentation; Map features to class corresponding values
                # sort by class index to ensure pixel burn order
                tr_features_tr = tr_features_tr.sort_values(f"{self.target_col}_")
                # extract geometry and class index
                fr = list(
                    tr_features_tr[['geometry', f"{self.target_col}_"]].itertuples(index=False, name=None)
                )
            if len(fr) == 0:
                target = np.zeros(patch["shape"], dtype=np.int32)
            else:
                target = rasterio.features.rasterize(
                    fr, out_shape=patch["shape"],
                    all_touched=self.rasterize_all_touched, fill=0, transform=patch["ori_transform"]
                )

            # Create mask for the area where we have labels, particularly important for allow_partial_patches = True
            area = self.labeled_areas.loc[[patch.area_id]]  # Returns a dataframe
            area_tr = area.to_crs(patch['ori_crs'])
            mask = rasterio.features.rasterize(
                area_tr['geometry'].to_list(), out_shape=patch["shape"],
                all_touched=self.rasterize_all_touched, fill=0, transform=patch["ori_transform"]
            )

            pt["target"] = torch.tensor(target, dtype=self.target_dtype)
            pt["target_mask"] = torch.tensor(mask, dtype=torch.bool)
            assert pt["target"].shape[-2:] == patch["shape"], f"{pt['target'].shape[-2:]}"
            assert pt['target_mask'].shape[-2:] == patch["shape"], f"{pt['target_mask'].shape[-2:]}"

        assert self.reference_source in pt, f"{self.reference_source in pt}"
        assert pt[self.reference_source].shape[-2:] == patch["shape"], f"{patch['shape']}"
        return pt

    # This overrides get_patch_raster_images in BaseDataset if there are extracted images. Otherwise it runs as before
    def get_patch_raster_images(self, patch):
        if self.extract_images_for_areas:
            return self.get_patch_raster_images_from_extracts(patch)
        else:  # Call the method from the baseDataset
            return super().get_patch_raster_images(patch)

    def get_patch_raster_images_from_extracts(self, patch):
        pt = {}
        as_im = self.assigned_images.query("area_id == @patch.area_id")
        as_im = as_im[as_im.intersects(patch.geometry)]

        for _, img in as_im.iterrows():
            src_name = img["src"]
            path = img[self.extracted_image_col]
            # transform to reference image crs
            patch_geom = patch.geometry
            ex_im = rasterio.open(path)
            wn_geom = geometry_to_crs(patch_geom, self.project_crs, ex_im.crs)
            # careful here, the windows is from the bounds not the geometry of the patch
            wn = rasterio.windows.from_bounds(*wn_geom.bounds, transform=ex_im.transform)
            # This is required due to a rounding error while creating the window
            wn = Window(np.floor(wn.col_off), np.floor(wn.row_off), round(wn.width), round(wn.height))
            img_arr = self.read_window_from_image(img_path=path, wn=wn, boundless=True)
            # need to reproject if different reference crs was used
            if ex_im.crs != patch["ori_crs"]:
                # get transform and shape of reprojection
                transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                    ex_im.crs, patch["ori_crs"], *img_arr.shape[1:], *wn_geom.bounds
                )
                dst = np.zeros((img_arr.shape[0], dst_width, dst_height), dtype=img_arr.dtype)
                dst, _ = rasterio.warp.reproject(
                    source=img_arr,
                    destination=dst,
                    src_transform=ex_im.transform,
                    src_crs=ex_im.crs,
                    dst_transform=transform,
                    dst_crs=patch["ori_crs"],
                    resampling=Resampling.nearest,
                )
                # center image to same size
                dst_x_off = dst.shape[1] - patch["shape"][0]
                dst_y_off = dst.shape[2] - patch["shape"][1]
                dst = dst[:, dst_x_off // 2: -int(np.ceil(dst_x_off / 2)), dst_y_off // 2: -int(np.ceil(dst_y_off / 2))]
                img_arr = dst

            img_arr = img_arr.astype(np.float32)
            nodataval = ex_im.nodatavals[0]  # might break
            nan_mask = np.isnan(img_arr) | (img_arr == nodataval)
            img_arr[nan_mask] = np.nan
            if src_name in pt:
                if self.merge_mode == "keep_first":
                    nan_mask = np.isnan(pt[src_name])
                    pt[src_name] = np.where(nan_mask, img_arr, pt[src_name])
                if self.merge_mode == 'keep_last':
                    pt[src_name] = np.where(nan_mask, pt[src_name], img_arr)
                if self.merge_mode == 'keep_biggest': # for patch grid that are located on the edge of one tile
                    first_mask = np.isnan(pt[src_name]) | (pt[src_name] == 0)
                    last_mask = np.isnan(img_arr) | (img_arr == nodataval) | (img_arr == 0)
                    if first_mask.sum() <= last_mask.sum():
                        pt[src_name] = np.where(nan_mask, img_arr, pt[src_name])
                    else:
                        pt[src_name] = np.where(nan_mask, pt[src_name], img_arr)
            else:
                pt[src_name] = img_arr

        for src_name in as_im["src"].unique():
            if src_name in pt:
                pt[src_name] = torch.tensor(pt[src_name], dtype=torch.float32)

        return pt

    def build_labeled_areas_table(self, labeled_areas):
        """
        Sample table:
        area_id, geometry, area_size_pixel ,polygons, patch_start_index, patch_end_index, overlapping_images
        1,  [[1,2], [300,400]], (5000,5000), [[[1,2], [5,6], [64,3]], [[83,45], [34, 45], [97, 66]]], 0, 120, [img1.tif, img2.tif]
        """

        labeled_areas, assigned_images = self.build_labeled_areas_table_(labeled_areas)
        if self.extract_images_for_areas and self.processed_dir is not None:
            ep = self.processed_dir / 'extracted_images'
            if ep.exists() and ep.iterdir():
                raise Exception(
                    f"Extracted images already exist in {ep}!!" +
                    f"\nPlease remove the folder to re-extract." +
                    f"\nOtherwise change image source to the extracted location and set image extract to False")
            else:
                ep.mkdir(exist_ok=True)
                logging.info(f"Extracting relevant image parts to {ep}")
                assigned_images = extract_images(areas=labeled_areas, images_df=self.images_df, base_dir=ep,
                                                 extracted_image_col=self.extracted_image_col,
                                                 assigned_images=assigned_images, prefix=self.dataset_name)
                assert self.extracted_image_col in assigned_images.columns
        else:
            logging.info("Not extracting images.")
        return labeled_areas, assigned_images

    def build_patch_table_sequentially(self, patch_start_index=0):
        patches = []
        height, width = self.patch_size
        images_grouped_by_area = self.assigned_images.query(f"src == '{self.reference_source}'").groupby('area_id')
        for area_idx, area_images in tqdm(images_grouped_by_area, desc="Iterating over areas"):
            area = self.labeled_areas.loc[area_idx]  # Returns a series
            img_indices = area_images.reset_index()["image_id"].values
            imgs = self.images_df.loc[img_indices]

            ori_crs, ori_transform, union_geometry = self.get_imgs_reference(imgs)
            # We still want to work with the overlapping geometry instead of the pure image geometry
            union_geometry = union_geometry.intersection(area.geometry)

            # Now we divide this geometry into smaller patches
            # Use the point approach; start with the top left point,
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
                }

                patches.append(patch)

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
            "rasterize_all_touched": config.get("rasterize_all_touched")
        }
        options.update(kwargs)
        return cls(labeled_areas=labeled_areas,
                   images_df=images_df,
                   target_features=target_features,
                   **options)


def get_patch_features(assigned_features, geometry, area_id=None):
    if area_id is not None:
        assigned_features = assigned_features.query(f"area_id == {area_id}")  # Target features
    assigned_features = assigned_features[assigned_features.intersects(geometry)]
    return assigned_features
