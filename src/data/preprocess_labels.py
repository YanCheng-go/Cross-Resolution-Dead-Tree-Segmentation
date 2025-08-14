""""reprocess labels and aois.
1. convert to WGS84
2. multipolygon to single polygon
3. remove empty labels and aois
4. remove null geometries
5. combine multiple gpkg to one for aois and labels independently
6. remove empty labels
"""

import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from pygeos import box
from tqdm import tqdm


def reprocess_labels(fp, layer_name=None):
    if fp.endswith(".gpkg"):
        assert layer_name is not None, "Layer name must be provided for gpkg files"
        try:
            gdf = gpd.read_file(fp, layer=layer_name)
        except:
            return
    elif fp.endswith(".geojson"):
        try:
            gdf = gpd.read_file(fp)
        except:
            return

    # convert to WGS84
    gdf = gdf.to_crs("EPSG:4326")
    # Explode multipart geometries into singlepart
    # This works for MultiPolygon, MultiLineString, and MultiPoint
    gdf = gdf.explode(index_parts=False)
    gdf.reset_index(drop=True, inplace=True)  # Reset index if necessary
    # remove empty labels
    gdf = gdf[~gdf.is_empty]
    # remove null geometries
    gdf = gdf[~gdf.isna()]
    # add a column indicating the filepath and another field for the filename
    gdf["location"] = fp
    gdf["filename"] = os.path.basename(fp)
    return gdf


def merge_vectors(vector_list, out_fp):
    # combine multiple gpkg to one for aois and labels independently
    if len(vector_list) == 0:
        return
    if len(vector_list) == 1:
        return vector_list[0]
    gdf = pd.concat(vector_list)
    gdf.reset_index(drop=True, inplace=True)
    gdf.to_file(out_fp, driver="GPKG")
    return gdf


def main(in_dir, label_name, aoi_name, aoi_out_fp, label_out_fp):

    gpkg_fp_list = list(Path(in_dir).rglob("*.gpkg"))

    # reprocess labels
    labels_list  = [reprocess_labels(fp, label_name) for fp in gpkg_fp_list]
    labels = merge_vectors(labels_list, label_out_fp)

    # reprocess aois
    aois_list = [reprocess_labels(fp, aoi_name) for fp in gpkg_fp_list]
    aois = merge_vectors(aois_list, aoi_out_fp)

    return aois, labels

def get_img_bbox(img_fp, out_crs='EPSG:4326', exts=[".tif", ".tiff"]):
    """Get the bounding box of an image tile and save as a polygon"""
    if not any([img_fp.endswith(e) for e in exts]):
        raise ValueError("Image file must be a tif or tiff file")
    src = rasterio.open(img_fp)
    bbox = src.bounds
    bbox = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=src.crs)
    bbox_gdf = bbox_gdf.to_crs(out_crs)
    bbox_gdf["location"] = img_fp
    bbox_gdf["filename"] = os.path.basename(img_fp)
    return bbox_gdf

def get_aois(img_fps, out_crs='EPSG:4326', exts=[".tif", ".tiff"], out_fp='./tmp/labeled_areas.gpkg'):
    """Get the bounding boxes of all images"""
    bbox_list = [get_img_bbox(fp, out_crs, exts) for fp in tqdm(img_fps)]
    bbox_gdf = pd.concat(bbox_list)
    bbox_gdf.reset_index(drop=True, inplace=True)
    bbox_gdf.to_file(out_fp, driver="GPKG")
    return bbox_gdf


if __name__ == '__main__':
    clemens_drone_datasets = False
    samuli_aerial_datasets = False

    # Reprocess labels and images from clemens's drone datasets
    if clemens_drone_datasets:
        in_dir = "/mnt/raid5/DL_TreeHealth_Aerial/ortho_labels/instance_segmenation_labels_clemens/labels_and_aois"
        label_name = "standing_deadwood"
        aoi_name = "aoi"
        aoi_out_fp = "/mnt/raid5/DL_TreeHealth_Aerial/ortho_labels/instance_segmenation_labels_clemens/aois_merged_test.gpkg"
        label_out_fp = "/mnt/raid5/DL_TreeHealth_Aerial/ortho_labels/instance_segmenation_labels_clemens/labels_merged_test.gpkg"
        main(in_dir, label_name, aoi_name, aoi_out_fp, label_out_fp)

    # reprocess labels and aois from samuli's aerial datasets
    if samuli_aerial_datasets:
        # reprocess aois
        with open('/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/deadtrees_images_20241031.txt') as f:
            img_fps = f.readlines()
        img_fps = [fp.strip() for fp in img_fps]
        get_aois(img_fps, out_fp='/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/deadtrees_images_20241031.gpkg')

        # reprocess labels
        with open('/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/deadtrees_labels_20241031.txt') as f:
            gpkg_fp_list = f.readlines()
        gpkg_fp_list = [fp.strip() for fp in gpkg_fp_list]
        labels_list  = [reprocess_labels(fp) for fp in gpkg_fp_list]
        labels = merge_vectors(labels_list, '/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/deadtrees_labels_20241031.gpkg')















