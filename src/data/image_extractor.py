import json
import logging
import os
import shutil
from pathlib import Path

import torch
import rasterio
from rasterio import mask, windows
from rasterio.windows import get_data_window
from scipy import stats
from tqdm import tqdm

from src.data.base_dataset import geometry_to_crs
import geopandas as gpd

import numpy as np

#https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)

def channel_description(img):
    d = stats.describe(img.flatten())
    return json.dumps(d._asdict(), separators=(',', ':'), cls=NumpyArrayEncoder)


def extract_images(areas: gpd.GeoDataFrame, images_df: gpd.GeoDataFrame,
                   base_dir: Path, extracted_image_col: str = 'extracted_path', assigned_images: gpd.GeoDataFrame = None,
                   prefix: str = 'train'):
    if images_df.index.name != 'image_id':
        images_df.index.name = 'image_id'
    if assigned_images is None:
        assigned_images = gpd.sjoin(
            images_df[["src", "geometry"]], areas[["geometry"]],
            predicate="intersects", how="inner"
        ).rename(columns={"index_right": "area_id"})
        assigned_images.set_index(['area_id'], append=True, inplace=True)
        assert assigned_images.index.names == ["image_id", "area_id"]

    areas_grouped_by_images = assigned_images.groupby('image_id')
    # For each input image, get all training areas in the image
    extracted_dict = {}
    cnt = 0

    os.makedirs(base_dir.parents[0] / 'extracted_images_masked', exist_ok=True)

    for img_idx, area_images in areas_grouped_by_images:
        img = images_df.loc[img_idx]
        im_path = img['path']
        area_indices = area_images.reset_index()["area_id"].values
        # For each area, extract the image channels and write img and annotation channels to a merged file
        for area_id in tqdm(area_indices, f"Extracting areas for {os.path.basename(im_path)}", position=0):
            ag = areas['geometry'].loc[area_id]
            agt = geometry_to_crs(ag, areas.crs, img['ori_crs'])
            # Extract the part of input image that overlaps training area, with optional resampling

            ro = rasterio.open(im_path)
            wn = rasterio.windows.from_bounds(*agt.bounds, img['ori_transform'])
            wn = rasterio.windows.Window(np.floor(wn.col_off), np.floor(wn.row_off), round(wn.width),
                                         round(wn.height))

            # A quick walk around for images smaller than labeled areas...
            try:
                wn = windows.intersection(wn, get_data_window(ro))
            except:
                continue

            win_transform = ro.window_transform(wn)
            cnt_new = f'{img_idx}_{area_id}'
            output_fp = base_dir / (f"{cnt_new}.tif" if len(prefix) == 0 else f"{prefix}_{cnt_new}.tif")
            cnt += 1
            rimg = ro.read(window=wn, boundless=False)
            logging.info(f"Read image mean: {rimg.mean()}")
            # logging.info
            if rimg.mean() == 0:
                logging.info(f"Extracted image has 0 mean!! The image is likely to be corrupted/useless!")
            print(f"No datavals = {ro.nodatavals}")

            ndt = None
            if isinstance(ro.nodatavals, list):
                ndt = ro.nodatavals[0]

            if rimg.shape[1] > 0 and rimg.shape[2] > 0:
                # For difference in band and pixel interleaving see this:
                # https: // gdal.org / development / rfc / rfc14_imagestructure.html
                # https://www.loc.gov/preservation/digital/formats/fdd/fdd000305.shtml
                # https://www.l3harrisgeospatial.com/Learn/Blogs/Blog-Details/ArtMID/10198/ArticleID/15508/Pixel-Interleave-%E2%80%93-Why-You-Should-Care-and-How-To-Handle-It
                # Pixel or "Band interleaving by pixel" (BIP) should be faster for window read as all bands of a pixel are together
                with rasterio.open(output_fp,
                                   mode='w',
                                   driver='GTiff',
                                   height=rimg.shape[1],
                                   width=rimg.shape[2],
                                   count=rimg.shape[0],
                                   dtype=ro.profile['dtype'],
                                   crs=ro.crs,
                                   transform=win_transform,
                                   tiled=False,
                                   interleave='pixel',
                                   nodata=ndt
                                   ) as dst:

                    dst.write(rimg)
                    for c in range(rimg.shape[0]):
                        dst.set_band_description(c + 1, channel_description(rimg[c]))

                # Mask with multipolygons
                masked_fp = base_dir.parents[0] / 'extracted_images_masked' / os.path.basename(output_fp)
                if ag.geom_type == 'MultiPolygon':
                    with rasterio.open(output_fp) as src:
                        ori_meta = src.meta
                        out_image, out_transform = rasterio.mask.mask(ro, list(agt), filled=True)
                        ori_meta.update({"height": out_image.shape[1],
                                         "width": out_image.shape[2]})

                    # A quick walk around the empty images generated for some reasons
                    if not np.sum(out_image) == 0 or np.sum(np.isnan(out_image)) == out_image.size:
                        with rasterio.open(masked_fp, "w", **ori_meta) as dst:
                            dst.write(out_image)
                else:
                    shutil.copyfile(output_fp, masked_fp)

                extracted_dict[(img_idx, area_id)] = output_fp  # Run it again for the ones not saving output_fp
    assigned_images[extracted_image_col] = assigned_images.index.map(extracted_dict)
    assigned_images.dropna(inplace=True)
    return assigned_images


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-df", type=Path, help="Path to the images dataframe stored as geopackage or pt file")
    parser.add_argument("-a", "--areas", type=Path, help="Path to the geopackage/shapefile containing the areas")
    parser.add_argument("-o", "--output", type=Path, help="Path to the output")
    parser.add_argument("-p", "--prefix", type=str, help="Prefix for the output files", default="")
    args = parser.parse_args()

    areas = gpd.read_file(args.areas)
    if args.image_df.suffix == ".gpkg":
        images_df = gpd.read_file(args.image_df)
    elif args.image_df.suffix == ".pt":
        images_df = torch.load(args.image_df)
    else:
        raise ValueError("Unknown file format for image df")
    if args.output is None:
        output_dir = args.areas.name.replace(".gpkg", "").replace(".pt", "")
    else:
        output_dir = args.output
    extract_images(areas, images_df, output_dir, None, None, args.prefix)