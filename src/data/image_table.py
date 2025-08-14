import json
import logging
import os
import time
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box
from torch import save, load
from tqdm import tqdm

from src.data.base_dataset import get_df_in_single_crs
from src.modelling import helper
from src.utils.data_utils import is_same_crs


def read_file_meta(file, src_name, src):
    # Reads the meta information for a given file
    im_dict = {
        'name': file.split(os.sep)[-1],
        'path': file,
        'src': src_name,
        'image_file_type': src['image_file_type'],
        'image_file_prefix': src['image_file_prefix'],
        'image_file_postfix': src['image_file_postfix']
    }
    try:
        img = rasterio.open(file)
    except rasterio.errors.RasterioIOError:
        return file, False, None

    nodata = np.array(img.nodatavals)
    for i in range(len(nodata)):
        if nodata[i] is None:
            nodata[i] = np.nan
    nodata = nodata.astype(np.float32).reshape((-1, 1, 1))

    gm = box(*img.bounds)
    im_dict.update({
        'ori_geometry': gm,
        'geometry': gm,  # !! Placeholder which is transformed to project crs
        'shape': img.shape,
        'profile': img.profile,
        'ori_transform': img.transform,
        'resolution_per_pixel': img.res,
        'indexes': img.indexes,
        'dtypes': img.dtypes,
        'nodatavals': nodata,
        'ori_crs': img.crs
    })

    img.close()
    return file, True, im_dict


def build_images_table(
        image_srcs: dict, reference_source: str, project_crs, save_idf=False,
        processed_dir=None, n_jobs=1, image_df_name='image_df.pt'
):
    """
        Pass cpu 'all' or -1 for using all of them, N for one and 0/1 for single thread
    """

    save_idf = save_idf and processed_dir is not None
    processed_dir = Path(processed_dir) if save_idf else None
    if save_idf and (processed_dir / image_df_name).is_file():
        logging.info(f"Reading df from {processed_dir / image_df_name}")
        processed_dir.mkdir(exist_ok=True)
        return load(processed_dir / image_df_name)
    else:
        logging.info("Creating image df now.")

    all_images = []
    crs_srcs = {}
    cpu_count = helper.get_cpu_count(n_jobs)
    unreadable_files = []

    for src_name, src in image_srcs.items():
        if 'recursive' not in src:  # For backward compatibility
            recursive = False
        else:
            recursive = src['recursive']
        filelist_path = src.get('filelist_path', None)
        if filelist_path is None:
            files = glob(
                f"{src['base_path']}/{src['image_file_prefix']}*{src['image_file_postfix']}{src['image_file_type']}",
                recursive=recursive
            )
        else:
            with open(filelist_path, 'r') as file:
                files = file.read().split('\n')
        if cpu_count == 0 or cpu_count == 1:
            file_meta = [read_file_meta(file, src_name, src) for file in tqdm(files)]
        else:
            time1 = time.time()
            from joblib import Parallel, delayed
            file_meta = Parallel(n_jobs=cpu_count)(delayed(read_file_meta)(file, src_name, src) for file in files)
            time2 = time.time()
            logging.info(f"Read {len(file_meta)} files in {time2 - time1} seconds")

        for (fp, readable, meta) in file_meta:
            if readable:
                all_images.append(meta)

                if src_name not in crs_srcs:
                    crs_srcs[src_name] = set([meta['ori_crs']])
                else:
                    crs_srcs[src_name].add(meta['ori_crs'])
            else:
                unreadable_files.append(fp)

    if len(unreadable_files) > 0:
        logging.info(f"WARNING: following files couldn't be read:\n {unreadable_files}")
    if not all_images:
        raise ValueError('No images available! Please provide the correct path')
    if project_crs is None:
        Warning('The project CRS is not set! Setting it to reference source crs')
        project_crs = list(crs_srcs[reference_source])[0]

    different_crs = False
    for k, v in crs_srcs.items():
        if len(v) > 1:
            different_crs = True
            logging.info(f"IFO: src {k} has multiple crs {v}, will project to project crs")
            break
    temp_crs = list(crs_srcs[reference_source])[0]
    idf = gpd.GeoDataFrame(all_images,
                           crs=temp_crs)  # Set to first crs of the reference source (there can be potentially many)
    idf.index.set_names(['image_id'], inplace=True)

    # convert to project crs if different crs exist
    if different_crs or not is_same_crs(temp_crs, project_crs):
        idf = get_df_in_single_crs(idf, project_crs)

    # save df if processed_dir is set and save_idf is True
    if save_idf and not (processed_dir / image_df_name).is_file():
        processed_dir.mkdir(exist_ok=True)
        save(idf, processed_dir / image_df_name)

    return idf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_srcs', type=str, required=True, help='json file with image sources')
    parser.add_argument('-r', '--reference_source', type=str, required=False, help='reference source, if not set, first source is used')
    parser.add_argument('-c', '--project_crs', type=str, help='project crs', default='EPSG:4326')
    parser.add_argument('-s', '--save_idf', action='store_true', help='save image df', default=True)
    parser.add_argument('-p', '--processed_dir', type=str, help='processed dir', default='.')
    parser.add_argument('--cpu', type=int, help='cpu count', default=1)
    args = parser.parse_args()

    image_srcs = json.load(open(args.image_srcs))
    rf_source = args.reference_source if args.reference_source is not None else list(image_srcs.keys())[0]
    image_df_name = "_".join(image_srcs.keys())
    image_df = build_images_table(
        image_srcs, rf_source, args.project_crs, args.save_idf, args.processed_dir, args.cpu, f"{image_df_name}.pt"
    )
    wp = os.path.join(args.processed_dir, f"{image_df_name}.gpkg")
    image_df[['geometry', 'name', 'path']].to_file(wp, driver='GPKG')
    logging.info(f"Saved image df to {wp}")
