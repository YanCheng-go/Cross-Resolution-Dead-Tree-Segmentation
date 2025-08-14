# Read the footprints, read the image to correct and the reference image.
import argparse
import os
import logging
import numpy as np
import rasterio
import rasterio.features
from pathlib import Path

from skimage.filters import sobel_h, sobel_v
from torch import save, load
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import box, LineString, Polygon
from scipy.ndimage import gaussian_filter

from src.utils.data_utils import histogram_match, write_geotiff_to_file
from numba import njit
@njit(parallel=False)
def rasterize_line_string(rows:np.array, cols:np.array, output:np.array, row_max:int, col_max:int):
    row_min = 0
    col_min = 0
    # Bresentham line rasterization algorithm
    for i in range(len(rows) - 1):
        r0, c0 = rows[i], cols[i]
        r1, c1 = rows[i + 1], cols[i + 1]
        if (r0<row_min and r1<row_min) or (r0>row_max and r1>row_max) or (c0<col_min and c1<col_min) or (c0>col_max and c1>col_max) or (r1 == r0 and c1 == c0):
            continue
        dr = r1 - r0
        dc = c1 - c0
        ln = np.sqrt(dr ** 2 + dc ** 2)
        d = max(abs(dr), abs(dc))
        dru = dr / ln
        dcu = dc / ln
        dr = dr / d
        dc = dc / d
        for j in range(d):
            nr = int(r0 + j * dr)
            nc = int(c0 + j * dc)
            if nr > row_min and nr < row_max and nc > col_min and nc < col_max:
                output[nr, nc, 0] = -dcu # Store the orthaogonal vector
                output[nr, nc, 1] = dru
    return output

def score_cutlines(r_inp_img: np.array, cutlines_in_inp_crs: gpd.GeoDataFrame, transform = None, visualize=False):

    cutlines_score = {}

    gr_x = sobel_h(r_inp_img) # Gradient along the rows (up - down)
    gr_y = sobel_v(r_inp_img) # Gradient along the cols (left - right)
    gr = np.stack([gr_x, gr_y], axis=-1) # Stack the gradients
    print(cutlines_in_inp_crs)
    all_score = np.zeros_like(r_inp_img, dtype=np.float32)
    for rowindex, _ in tqdm(cutlines_in_inp_crs.iterrows()):
        line = cutlines_in_inp_crs.loc[rowindex, 'geometry']
        if type(line) == Polygon:
            xs, ys = line.boundary.xy
        elif type(line) == LineString:
            xs, ys = line.xy
        else:
            print(f"Unknown datatype {type(line)} encountered.")
            print(line)
            continue
            # raise Exception(f"Unknown datatype {type(line)} encountered.")
        if transform is not None:
            rows, cols = rasterio.transform.rowcol(transform, xs, ys )
        else:
            rows, cols = xs, ys
        a = np.zeros(list(r_inp_img.shape) + [2], dtype=np.float32)
        row_max, col_max = r_inp_img.shape
        rast_line = rasterize_line_string(rows=rows, cols=cols, output=a, row_max=row_max, col_max=col_max)
        # Compute the dot product between the gradient and the line

        dot_prod = np.abs(np.sum(gr * rast_line, axis=-1))

        if visualize:
            print(rows, cols)
            import matplotlib.pyplot as plt
            plt.subplot(1, 4, 1)
            plt.imshow(r_inp_img, cmap='gray')
            plt.subplot(1, 4, 2)
            plt.imshow(rast_line[..., 0], cmap='gray')
            plt.subplot(1, 4, 3)
            plt.imshow(rast_line[..., 1], cmap='gray')
            plt.subplot(1, 4, 4)
            plt.imshow(dot_prod, cmap='gray')
            plt.show()
        all_score += dot_prod
        cutlines_score[rowindex] = np.abs(np.mean(dot_prod))

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(all_score, cmap='gray')
        plt.show()
    print(cutlines_score)
    cutlines_in_inp_crs['score'] = cutlines_in_inp_crs.index.map(cutlines_score)
    return cutlines_in_inp_crs

def histogram_match_images(inp_image: rasterio.DatasetReader, ref_image: rasterio.DatasetReader,
                           cutlines: gpd.GeoDataFrame, dir_hist=None, **kwargs):

    r_inp_img = inp_image.read()
    r_inp_img = r_inp_img[[0]]
    r_ref_img = ref_image.read()
    r_ref_img = r_ref_img[[0]]  #np.mean(r_ref_img[:3], axis=0, keepdims=True) # Convert to RGB for
    cutlines_in_inp_crs = cutlines.to_crs(inp_image.crs)
    cutlines_in_ref_crs = cutlines.to_crs(ref_image.crs)

    output_image = np.zeros(inp_image.shape, dtype=np.float32)

    for rowindex, _ in tqdm(cutlines_in_inp_crs.iterrows()):
        s_i = cutlines_in_inp_crs.loc[rowindex].geometry
        valid_mask_inp = rasterio.features.rasterize([s_i], out_shape=inp_image.shape,
                                                     transform=inp_image.transform,
                                                     fill=0, dtype=rasterio.uint8, all_touched=True)
        r_i = cutlines_in_ref_crs.loc[rowindex].geometry
        valid_mask_ref = rasterio.features.rasterize([r_i], out_shape=ref_image.shape,
                                                     transform=ref_image.transform,
                                                     fill=0, dtype=rasterio.uint8, all_touched=True)

        assert np.sum(valid_mask_inp) != 0
        assert np.sum(valid_mask_ref) != 0

        im = r_inp_img * valid_mask_inp
        rf = r_ref_img * valid_mask_ref

        fl = dir_hist / f"{rowindex}_mask_inp.tif"
        write_geotiff_to_file(count=1, height=valid_mask_inp.shape[0], width=valid_mask_inp.shape[1],
                              bounds=inp_image.bounds, file_path=fl, crs='EPSG:4326',
                              data=np.expand_dims(valid_mask_inp, axis=0),
                              dtype=rasterio.uint8)

        fl = dir_hist / f"{rowindex}_mask_ref.tif"
        write_geotiff_to_file(count=1, height=valid_mask_ref.shape[0], width=valid_mask_ref.shape[1],
                              bounds=ref_image.bounds, file_path=fl, crs='EPSG:4326',
                              data=np.expand_dims(valid_mask_ref, axis=0),
                              dtype=rasterio.uint8)
        out_hist = histogram_match(im, rf, nodata_val=0, plot_filename=dir_hist / f"{rowindex}_histogram.png", **kwargs)
        output_image = np.where(valid_mask_inp, out_hist, output_image)
    return output_image


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inp_image", type=str, required=True, help="Input image")
    argparser.add_argument("--ref_image", type=str, required=True, help="Reference image")
    argparser.add_argument("--cutlines", type=str, required=True, help="Cutlines")
    argparser.add_argument("--output_image", type=str, required=True, help="Output image")
    args = argparser.parse_args()

    inp_image = rasterio.open(args.inp_image)
    ref_image = rasterio.open(args.ref_image)
    cutlines = gpd.read_file(args.cutlines)

    inp_img_bx = box(*inp_image.bounds)

    cutlines_inp_crs = cutlines.to_crs(inp_image.crs)
    cutlines_inp_crs = cutlines_inp_crs[cutlines_inp_crs.intersects(inp_img_bx)] # Only keep cutlines that intersect with the input image
    cutlines_inp_crs = cutlines_inp_crs.clip(inp_img_bx) # Clip to inp image bounds
    dir_hist = Path("temp_histograms")
    dir_hist.mkdir(exist_ok=True)

    r_inp_img = inp_image.read()
    r_inp_img = r_inp_img[0]

    scored_cutlines = score_cutlines(r_inp_img, cutlines_inp_crs, transform=inp_image.transform, visualize=True)
    import ipdb; ipdb.set_trace()
    cutlines_inp_crs.to_file(dir_hist / "cutlines_inp_crs.gpkg", driver="GPKG")
    scored_cutlines.to_file(dir_hist / "scored_cutlines.gpkg", driver="GPKG")


    #filters = [
    #    {},
    #    {"filter": "gaussian", "sigma": 0.5},
    #    {"filter": "gaussian", "sigma": 1},
    #    {"filter": "gaussian", "sigma": 10},
    #    {"filter": "linspace_cutoff", "cutoff": 0.02},
    #    {"filter": "linspace_cutoff", "cutoff": 0.05},
    #]
    #for filter in filters:
    #    sub_dir_hist = dir_hist / f"{filter.get('filter', 'none')}_{filter.get('sigma', 0)}_{filter.get('cutoff', 0)}"
    #    sub_dir_hist.mkdir(exist_ok=True)

    #    out_image = histogram_match_images(inp_image, ref_image, cutlines_inp_crs, dir_hist=sub_dir_hist, **filter)

    #    rasterio.open(sub_dir_hist / args.output_image, 'w', driver='GTiff', height=out_image.shape[1], width=out_image.shape[2],
    #                  count=out_image.shape[0], dtype=out_image.dtype, crs=inp_image.crs, compress='lzw',
    #                  transform=inp_image.transform).write(out_image)
