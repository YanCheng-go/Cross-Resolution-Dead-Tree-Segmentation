"""generic utils for processing input images and training labels, more specific pre-processing pipelines for individual
 datasets can be found in src/data/[area_name].py"""
import multiprocessing
import os
import re
import shutil
from rasterio.crs import CRS
import sys
import resource
from functools import partial
from pathlib import Path
from shutil import move

import geopandas as gpd

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from osgeo import gdal
from rasterio.enums import ColorInterp
from tqdm import tqdm

from joblib import Parallel, delayed
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from train.base import init_config
from train.treehealth_ordinal_watershed import THOWatershedTrainer

import math

from rasterio.warp import calculate_default_transform, reproject, Resampling
import scienceplots
plt.style.use(['nature', 'science'])


def retrieve_image_info(image_fps_path, image_info_path):
    with open(image_fps_path) as f:
        img_fps = f.readlines()
    img_fps = [Path(fp.strip()) for fp in img_fps]
    with open(image_info_path, 'w') as f:
        f.write('filename,folder,crs,resolution,dtype,nodata,count,driver,width,height\n')
        for img_fp in tqdm(img_fps, desc='Getting image info', unit='image', leave=False):
            with rasterio.open(img_fp) as src:
                fn = os.path.basename(img_fp)
                dir = os.path.dirname(img_fp)
                crs = src.crs
                res = src.res[0]
                dtp = src.dtypes[0]
                ndata = src.nodata
                count = src.count
                driver = src.driver
                width = src.width
                height = src.height
            f.write(f'{fn},{dir},{crs},{res},{dtp},{ndata},{count},{driver},{width},{height}\n')


def get_extent(raster_fp):
    src = gdal.Open(raster_fp)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    sizeX = src.RasterXSize * xres
    sizeY = src.RasterYSize * yres
    lrx = ulx + sizeX
    lry = uly + sizeY
    # format the extent coords
    extent = '{0} {1} {2} {3}'.format(ulx, lry, lrx, uly)
    return extent


def merge_from_vrt(dst, vrt):
    """Merge tiles in vrt file to a tif file"""
    gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
    gdal.Translate(dst, vrt, format='GTiff',
                   creationOptions='-co "BIGTIFF=YES" -co "NUM_THREADS=ALL_CPUS"',
                   callback=gdal.TermProgress_nocb)


def max_merge_vrt(n_bands, vrt_fp, out_fp):
    """Add maximum composite function in vrt file"""
    with open(vrt_fp, "r") as myfile:
        s = myfile.read()

    for band_id in range(1, n_bands + 1):
        replace = f'''
<VRTRasterBand dataType="Float32" band="{band_id}" subClass="VRTDerivedRasterBand">

    <PixelFunctionType>max</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[
import numpy as np
def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,
             raster_ysize, buf_radius, gt, **kwargs):
             out_ar[:] = np.ma.max(in_ar, axis = 0, fill_value=0)
]]>
    </PixelFunctionCode>'''
        find = f'''<VRTRasterBand dataType="Float32" band="{band_id}">'''
        s = re.sub(find, replace, s)

    with open(out_fp, 'w') as filetowrite:
        filetowrite.write(s)


# out_ar[:] = np.nanmax(in_ar, axis = 0)
def build_raster_vrt_by_crs(fps, out_dir, prefix='output_', crs_list=['EPSG:26910', 'EPSG:26911']):
    out_fps = [os.path.join(out_dir, prefix + f"{crs.replace(':', '')}.vrt") for crs in crs_list]
    in_fps_list = [[fp for fp in fps if rasterio.open(fp).crs == crs] for crs in crs_list]
    params_list = list(zip(in_fps_list, out_fps))
    list(build_raster_vrt(*params) for params in params_list)
    return out_fps


def build_raster_vrt(in_fps, out_fp):
    if os.path.exists(out_fp):
        return
    options = dict(
        allowProjectionDifference=True,
        # targetAlignedPixels=True,
        callback=gdal_progress_callback,
        # callback_data=tqdm(leave=False, position=1)
    )
    gdal.BuildVRT(out_fp, in_fps, **options)


def gdal_progress_callback(complete, message, data):
    """Callback function to show progress during GDAL operations such gdal.Warp() or gdal.Translate().

    Expects a tqdm progressbar in 'data', which is passed as the 'callback_data' argument of the GDAL method.
    'complete' is passed by the GDAL methods, as a float from 0 to 1
    """
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def resample_raster(res_factor=10 / 20, out_res=None, fp_list=None,
                    in_dir='/mnt/erda/TreeMortDataPool/fromEliasMirela/Swissimage_Ortho_GR_new',
                    out_dir='/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/swiss_rgb_20cm', src_crs=None, out_crs=None,
                    compress=False):

    """# res_factor 25 / 60 (spain) # 1 / 3 (germany/denmark)"""

    os.makedirs(out_dir, exist_ok=True)

    if fp_list is None and in_dir is not None:
        fp_list = []
        for dirpath, _, filenames in os.walk(in_dir):
            for f in filenames:
                fp = os.path.abspath(os.path.join(dirpath, f))
                if fp.endswith('.tif') or fp.endswith('.jp2'):
                    fp_list.append(fp)

    for fp in tqdm(fp_list):
        output_fp = os.path.join(out_dir, os.path.basename(fp))
        input_fp = fp
        if not os.path.exists(output_fp):
            raster_copy(output_fp, input_fp, mode="warp", resample=res_factor, out_res=out_res,
                        out_crs=out_crs, src_crs=src_crs, bands=None, bounds=None, bounds_crs=None,
                        multi_core=True, pbar=None, compress=compress, cutline_fp=None, resample_alg=gdal.GRA_Bilinear)

    fp_list = [os.path.join(out_dir, i) + '\n' for i in os.listdir(out_dir) if
               i.endswith('.tif') or i.endswith('.jp2')]
    with open(os.path.join(out_dir, 'fp_list.txt'), 'w') as txtfile:
        txtfile.writelines(fp_list)


def raster_copy(output_fp, input_fp, mode="warp", resample=None, out_res=None, src_crs=None, out_crs=None, bands=None, bounds=None, bounds_crs=None,
                multi_core=False, pbar=None, compress=False, cutline_fp=None, resample_alg=gdal.GRA_Bilinear):
    """ Copy a raster using GDAL Warp or GDAL Translate, with various options.

    The use of Warp or Translate can be chosen with 'mode' parameter. GDAL.Warp allows full multiprocessing,
    whereas GDAL.Translate allows the selection of only certain bands to copy.
    A specific window to copy can be specified with 'bounds' and 'bounds_crs' parameters.
    Optional resampling with bi-linear interpolation is done if passed in as 'resample'!=1.
    """

    # Common options
    base_options = dict(
        creationOptions=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256", "BIGTIFF=IF_SAFER",
                         "NUM_THREADS=ALL_CPUS"],
        callback=gdal_progress_callback,
        callback_data=pbar
    )
    if compress:
        base_options["creationOptions"].append("COMPRESS=LZW")
    if resample != 1:
        # Get input pixel sizes
        raster = gdal.Open(input_fp)
        gt = raster.GetGeoTransform()
        x_res, y_res = gt[1], -gt[5]

        if out_res is not None:
            base_options["xRes"] = out_res
            base_options["yRes"] = out_res
        elif resample is not None:
            base_options["xRes"] = x_res / resample,
            base_options["yRes"] = y_res / resample,

        base_options["resampleAlg"] = resample_alg

    # # if there is a bug in relation to the projection system, better to make sure the input of src_crs
    # and out_crs, instead of using default values
    # if out_crs is None:
    #     out_crs = 'EPSG:{}'.format(re.findall(r'\d+', rasterio.open(input_fp).crs.to_string().split('AUTHORITY["EPSG",')[-1])[0])
    #     src_crs = out_crs

    # Use GDAL Warp
    if mode.lower() == "warp":
        warp_options = dict(
            srcSRS=src_crs,
            dstSRS=out_crs,
            cutlineDSName=cutline_fp,
            outputBounds=bounds,
            outputBoundsSRS=bounds_crs,
            multithread=multi_core,
            warpOptions=["NUM_THREADS=ALL_CPUS"] if multi_core else [],
            warpMemoryLimit=1000000000,  # processing chunk size. higher is not always better, around 1-4GB seems good
        )
        return gdal.Warp(output_fp, input_fp, **base_options, **warp_options)

    # Use GDAL Translate
    elif mode.lower() == "translate":
        translate_options = dict(
            bandList=bands,
            outputSRS=out_crs,
            projWin=[bounds[0], bounds[3], bounds[2], bounds[1]] if bounds is not None else None,
            projWinSRS=bounds_crs,
        )
        return gdal.Translate(output_fp, input_fp, **base_options, **translate_options)

    else:
        raise Exception("Invalid mode argument, supported modes are 'warp' or 'translate'.")


def get_driver_name(extension):
    """Get GDAL/OGR driver names from file extension"""
    if extension.lower().endswith("tif"):
        return "GTiff"
    elif extension.lower().endswith("jp2"):
        return "JP2OpenJPEG"
    elif extension.lower().endswith("shp"):
        return "ESRI Shapefile"
    elif extension.lower().endswith("gpkg"):
        return "GPKG"
    else:
        raise Exception(f"Unable to find driver for unsupported extension {extension}")


def memory_limit(percentage: float):
    """Set soft memory limit to a percentage of total available memory."""
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * percentage), -1))
    # print(f"Set memory limit to {int(percentage*100)}% : {get_memory() * percentage/1024/1024:.2f} GiB")


def get_memory():
    """Get available memory from linux system.

    NOTE: Including 'SwapFree:' also counts cache as available memory (so remove it to only count physical RAM).
    This can still cause OOM crashes with a memory-heavy single thread, as linux won't necessarily move it to cache...
    """
    with open('/proc/meminfo', 'r') as mem_info:
        free_memory = 0
        for line in mem_info:
            if str(line.split()[0]) in ('MemFree:', 'Buffers:', 'Cached:', 'SwapFree:'):
                free_memory += int(line.split()[1])
    return free_memory


def memory(percentage):
    """Decorator to limit memory of a python method to a percentage of available system memory"""
    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 / 1024
                print('Available memory: %.2f GB' % mem)
                sys.stderr.write('\n\nERROR: Memory Exception\n')
                sys.exit(1)
        return wrapper
    return decorator


def histogram_match(input_img, ref_img, nodata_val=0):
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

        # Get unique pixel values, their counts and indices
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

        # Linearly interpolate ref values between input and ref cumulative distribution functions
        interpolated_vals = np.interp(input_cdf, ref_cdf, ref_values)
        interpolated_vals = np.append(interpolated_vals, nodata_val)  # no-data values are in the last bin of input_idxs

        # Map interpolated values to pixels and reshape back to band shape
        out_img[band, ...] = interpolated_vals[input_idxs].reshape(input_band.shape)

    return out_img


def create_tile_index(out_fp, wd=None, fp_list=None, out_crs=None, mul_crs=None):

    if fp_list is None:
        txt_fp = os.path.join(wd, 'fp_list.txt')
        with open(txt_fp, 'w') as txtfile:
            ls = [os.path.join(wd, i) + '\n' for i in os.listdir(wd) if i.endswith('.tif') or i.endswith('.jp2')]
            txtfile.writelines(ls)

    if mul_crs is not None:
        for i in ls:
            for crs_ in mul_crs:
                if rasterio.open(i.split('\n')[0]).crs == crs_:
                    txt_fp = os.path.join(wd, 'fp_list_{}.txt'.format(crs_.split(':')[-1]))
                    with open(txt_fp, 'a') as txtfile:
                        txtfile.write(i)

        txt_fps = []
        for crs_ in mul_crs:
            txt_fp = os.path.join(wd, 'fp_list_{}.txt'.format(crs_.split(':')[-1]))
            txt_fps.append(txt_fp)

        for txt_fp in txt_fps:
            out_fp0 = os.path.splitext(out_fp)[0] + '_' + os.path.splitext(txt_fp)[0].split('fp_list_')[-1] + os.path.splitext(out_fp)[-1]
            if out_crs is None:
                os.system(f'gdaltindex -tileindex location {out_fp0} --optfile {txt_fp}')
            else:
                os.system(f'gdaltindex -tileindex location -t_srs {out_crs} {out_fp0} --optfile {txt_fp}')

    else:
        if out_crs is None:
            os.system(f'gdaltindex -tileindex location {out_fp} --optfile {txt_fp}')
        else:
            os.system(f'gdaltindex -tileindex location -t_srs {out_crs} {out_fp} --optfile {txt_fp}')


def add_pseudo_NIR(out_dir, in_dir=None, fp_txt=None, create_zeros=True, create_random=False):
    """Add pseudo NIR band as random numbers or zeros"""

    assert (in_dir!=None) or (fp_txt!=None)

    if in_dir is not None:
        filenames = os.listdir(in_dir)
        fp_list = [os.path.join(in_dir, f) for f in filenames]

    if fp_txt is not None:
        fp_list = []
        with open(fp_txt, 'r') as txtfile:
            for l in txtfile.readlines():
                fp_list.append(l.split('\n')[0])

    def process_file(full_path):
        f = os.path.basename(full_path)

        if not f.endswith(".tif"):
            return

        dr = rasterio.open(full_path)
        meta = dr.meta

        # if rgb, we create new NIR band. if NIR present, we replace that
        input_bands = meta["count"]
        assert input_bands == 3 or input_bands == 4, "number of input bands is unexpected"
        meta["count"] = 4

        # save for write
        colorinterp = dr.colorinterp

        # read data into memory
        data = dr.read()

        # simply ignore existing NIR band
        if input_bands == 4:
            data = data[:3]
            colorinterp = colorinterp[:3]

        if create_zeros:
            # add zeros on the fourth band
            zeros = np.zeros_like(data[0])
            zeros = np.expand_dims(zeros, axis=0)
            data_zeros = np.append(data, zeros, axis=0)
            os.makedirs(out_dir, exist_ok=True)
            with rasterio.open(os.path.join(out_dir, f), mode="w", **meta, compress="DEFLATE") as out:
                out.colorinterp = (*colorinterp, ColorInterp.undefined)
                out.write(data_zeros)

        if create_random:
            # add random values between 0 and 255 to fourth band
            rand = (np.random.rand(*data[0].shape) * 255).astype(np.uint8)
            rand = np.expand_dims(rand, axis=0)
            data_random = np.append(data, rand, axis=0)
            os.makedirs("/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/pseudo_NIR_random", exist_ok=True)
            with rasterio.open(os.path.join("/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/pseudo_NIR_random", f),
                               mode="w", **meta, compress="DEFLATE") as out:
                out.colorinterp = (*colorinterp, ColorInterp.undefined)
                out.write(data_random)

    Parallel(n_jobs=16)(delayed(process_file)(x) for x in tqdm(fp_list))


def export_tb_to_csv(log_dir, scalars, out_fp):
    """Export data from tensorboard to a csv file"""

    assert out_fp.endswith('.csv')
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    events = event_accumulator.Scalars(scalars)

    with open(out_fp, 'w') as f:
        f.write("step,value\n")
        # with tqdm(total=total_steps, desc=f'Exporting {os.path.basename(log_dir)}') as pbar:
        list(map(lambda scalar_event: f.write(f"{scalar_event.step},{scalar_event.value}\n"), tqdm(events)))

    return out_fp


def process_scalar(scalar_event):
    return f"{scalar_event.step},{scalar_event.value}\n"


def export_patch_resolutions(run_dir, output_dir):
    # Initialize EventAccumulator
    ea = EventAccumulator(run_dir)
    ea.Reload()

    # Extract scalar data for 'patch_resolutions'
    scalar_data = ea.Scalars('patch_resolutions')

    # Export to CSV
    csv_path = os.path.join(output_dir, f'{os.path.basename(run_dir)}.csv')
    with open(csv_path, 'w') as f:
        f.write("step,value\n")

        # Create a pool of processes
        with multiprocessing.Pool() as pool:
            # Use multiprocessing to process each scalar event and write to file
            for result in tqdm(pool.imap(process_scalar, scalar_data), total=len(scalar_data),
                               desc=f'Exporting {os.path.basename(run_dir)}'):
                f.write(result)


def extract_auto_split(config, save_pdf=False):
    """Extract patch_grids and the auto split indices"""
    Trainer = THOWatershedTrainer
    data, config = Trainer.init_data(config)
    train_dataset, val_dataset, test_dataset = Trainer.init_dataset_and_split(config, **data)
    patch_df = train_dataset.dataset.patch_df
    patch_df['auto_split'] = ''
    patch_df.iloc[train_dataset.indices.tolist(), -1] = 'train'
    patch_df.iloc[val_dataset.indices.tolist(), -1] = 'val'
    patch_df.iloc[test_dataset.indices.tolist(), -1] = 'test'
    if save_pdf:
        p = Path(config.processed_dir) / 'qgis' / 'patches'
        p.mkdir(exist_ok=True, parents=True)
        patch_df[['geometry', 'area_id', 'auto_split']].to_file(p / "patch_grid_autoSplit.gpkg", driver="GPKG")
    return patch_df

# Special zones for Svalbard and Norway
def getZones(longitude, latitude):
    if (latitude >= 72.0 and latitude < 84.0):
        if (longitude >= 0.0 and longitude < 9.0):
            return 31
    if (longitude >= 9.0 and longitude < 21.0):
        return 33
    if (longitude >= 21.0 and longitude < 33.0):
        return 35
    if (longitude >= 33.0 and longitude < 42.0):
        return 37
    return (math.floor((longitude + 180) / 6)) + 1


def findEPSG(longitude, latitude):
    zone = getZones(longitude, latitude)
    # zone = (math.floor((longitude + 180) / 6) ) + 1  # without special zones for Svalbard and Norway
    epsg_code = 32600
    epsg_code += int(zone)
    if (latitude < 0):  # South
        epsg_code += 100
    return epsg_code


def reproject_raster(out_dir, in_dir=None, fp_list=None, fp_txt=None, dst_crs='EPSG:4326', overwrite=False, driver=None,
                     compress=None):
    """reporject to another crs"""

    # retrieve the

    assert in_dir != None or fp_txt != None or fp_list != None

    os.makedirs(out_dir, exist_ok=True)

    img_fps = []
    if in_dir != None:
        img_fps.extend([os.path.join(in_dir, i) for i in os.listdir(in_dir) if i.endswith('.tif') or i.endswith('.jp2')])
    elif fp_list != None:
        with open(fp_list, 'r') as txtfile:
            for i in txtfile.readlines():
                img_fps.append(i.split('\n')[0])
    elif fp_txt != None:
        img_fps.extend(fp_txt)

    def main(fp):
        fn = os.path.basename(fp)
        out_fp = os.path.join(out_dir, fn)

        if not overwrite and os.path.exists(out_fp):
            return out_fp

        # import ipdb;ipdb.set_trace()
        with rasterio.open(fp) as src:
            transform, width, height = calculate_default_transform(
                src.crs, CRS.from_string(dst_crs), src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
            })

            dict_ = {'gtiff': 'tif', 'jp2': 'jp2', 'tiff': 'tif'}

            if 'driver' in kwargs and driver is not None:
                if kwargs['driver'].lower() != driver.lower():
                    kwargs.update(driver=driver)
                    out_fp = out_fp.replace(os.path.splitext(out_fp)[-1], f'.{dict_.get(driver.lower())}')
            if compress is not None:
                kwargs.update(compress=compress)

            with rasterio.open(out_fp, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                    )
        return out_fp

    out_fp_list = list(map(main, tqdm(img_fps)))
    return out_fp_list


def batch_to_raster(x, dataset_name='patch', out_dir='./tmp/pts_raster', crs=None, transform=None, overwrite=False):
    """Export a batch to a raster to the processed_dir with all bands in the batch"""

    patch_id, pt = x
    base_dir = Path(out_dir)
    base_dir.mkdir(exist_ok=True)
    cnt_new = '{}_{}'.format(dataset_name, patch_id)
    output_fp = base_dir / f"{cnt_new}.tif"
    ndt = None

    if os.path.exists(output_fp) and not overwrite:
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
                       transform=transform) as dst:
        dst.write(pt)


def add_crs_transform(in_dir, out_dir, ref_dir):
    fn_list = [i for i in os.listdir(in_dir)]
    ndt = None

    os.makedirs(out_dir, exist_ok=True)

    def f(fn):
        with rasterio.open(os.path.join(ref_dir, fn)) as src:
            crs = src.crs
            transform = src.transform
        with rasterio.open(os.path.join(in_dir, fn)) as src:
            pt = src.read()
        with rasterio.open(os.path.join(out_dir, fn),
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
                           transform=transform) as dst:
            dst.write(pt)

    list(map(f, tqdm(fn_list)))


def move_prediction(from_fp_list, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    with multiprocessing.Pool(processes=32) as pool:
        with tqdm(total=len(from_fp_list), desc='Move files', position=0, leave=True) as pb:
            for _, result in enumerate(pool.imap_unordered(partial(move, dst=to_dir), from_fp_list)):
                pb.update()


def analyses_patch_res(in_dir):
    """analyse the patch_resolutions scalar from tensorboard"""
    df = pd.read_csv(os.path.join(in_dir, 'run-.-tag-patch_resolutions_train.csv'))

    # Function to split every other character in a string
    def split_every_other(s):
        return [str(s)[i: i + 2] for i in range(len(str(int(s)))) if i % 2 == 0]

    # Apply the function to the 'Text' column
    df['Split'] = df['Value'].apply(split_every_other)

    # Convert the list of split characters into separate columns
    df_split = pd.DataFrame(df['Split'].to_list(), columns=[f'Char_{i+1}' for i in range(df['Split'].apply(len).max())])

    # Concatenate the original DataFrame with the split columns
    df = pd.concat([df, df_split], axis=1)

    # Drop the 'Split' column
    df.drop(columns=['Split'], inplace=True)
    df.to_csv(os.path.join(in_dir, 'patch_resolutions.csv'))

    return df


def data_distribution(data, col='patch_hectares', transform_func=lambda x: np.sqrt(x / 256 / 256 * 10000),
                      out_dir=None, prefix='', suffix='', bins=30, width=0.04, xlabel='image resolution (m)',
                      ylabel='number of patches\n(256 * 256 pixels)'):
    """Check the data distribution after using dataloader"""
    with plt.style.context('nature'):
        transform_func(data[col]).plot.hist(bins=bins, alpha=0.5, width=width, color='steelblue')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        # centerize the labels for bars in a histogram
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f'{prefix}data_distribution_{col}{suffix}.png'), dpi=300,
                        tight_layout=True)

        plt.show()


def get_masked_image_path_per_counties(in_dir: str):
    """ Get masked image path per counties  """
    # Create tile index for all rasters in a folder
    if not os.path.exists(Path(in_dir) / 'extracted_images_masked_prj' / 'tile_index.shp'):
        from postprocessing.util import create_tile_index
        create_tile_index(out_fp=Path(in_dir) / 'tile_index.shp', wd=Path(in_dir), out_crs='EPSG:4326')

    # Spatial join tile_index.shp and countries.shp
    # make sure all layers with the same crs
    if not os.path.exists(Path(in_dir) / 'image_path_county_name.shp'):
        gpd.sjoin(left_df=gpd.read_file(Path(in_dir) / 'extracted_images_masked_prj' / 'tile_index.shp'),
                  right_df=gpd.read_file('/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/WeightedSamping/'
                                         'countries.shp'),
                  how='inner', op='intersects').to_file(Path(in_dir) / 'image_path_county_name.shp')

    # Get the list of file paths per county
    gdf = gpd.read_file(Path(in_dir) / 'image_path_county_name.shp')

    for area_name in gdf['area_name'].unique():
        with open(Path(in_dir) / f'fp_list_{area_name}.txt', 'w') as f:
            for fp in gdf[gdf['area_name'] == area_name]['location'].tolist():
                fp = fp.replace('_prj', '')
                f.write(f'{fp}\n')


# retrerive mena and std. for each band for each country image sets
def get_mean_std_per_country(dataset_name: str, in_dir='/mnt/raid5/DL_TreeHealth_Aerial', band_sequence=(0, 1, 2, 3)):
    """Get mean and std for each band for each country image sets
    :param dataset_name: str, name of the dataset
    :param in_dir: str, input directory
    :param band_sequence: tuple, band indices in the order of RGBI"""

    # b0_mean,b0_std,b1_mean,b1_std,b2_mean,b2_std,b3_mean,b3_std
    if 'NAIP' in dataset_name:
        df = pd.read_csv(Path(in_dir) / f'{dataset_name}_global_stats_3.txt', sep=',')
    else:
        df = pd.read_csv(Path(in_dir) / f'{dataset_name}_global_stats.txt', sep=',')
    n_bands = len(df.columns) / 2
    band_sequence = band_sequence[:int(n_bands)]
    if 'swiss' in dataset_name.lower() and n_bands == 4:
        band_sequence = [1, 2, 3, 0]  # To check if the band sequence is correct

    if n_bands < 4:
        return ([round(df.iloc[0, i * 2], 4) for i in band_sequence] + [0.],
                [round(df.iloc[0, i * 2 + 1], 4) for i in band_sequence] + [0.])
    elif n_bands == 4:
        return ([round(df.iloc[0, i * 2], 4) for i in band_sequence],
                [round(df.iloc[0, i * 2 + 1], 4) for i in band_sequence])
    else:
        raise ValueError('The number of bands is not correct')


# Normalize rasters using mean and std. for each band
def normalize_raster(out_dir, mean_std_dict, area_name=None, in_dir=None, fp_list_txt=None):
    """Normalize rasters using mean and std. for each band"""
    assert in_dir != None or fp_list_txt != None
    os.makedirs(out_dir, exist_ok=True)

    if fp_list_txt is not None:
        fp_list = [i.split('\n')[0] for i in open(fp_list_txt, 'r').readlines()]
    if in_dir is not None:
        fp_list = [os.path.join(in_dir, fn) for fn in os.listdir(in_dir) if fn.endswith('.tif')]

    for fp in tqdm(fp_list):
        with rasterio.open(fp) as src:
            meta = src.meta
            data = src.read()
            new_data = []
            for i in range(data.shape[0]):
                new_data.append((data[i] - mean_std_dict['mean'][area_name][i]) / mean_std_dict['std.'][area_name][i])
            # concat list of array to 3D array
            data = np.stack(new_data, axis=0)
            out_fp = os.path.join(out_dir, os.path.basename(fp))
            meta.update(dtype='float32', compression='LWZ', count=src.count, nodata=0, driver='GTiff')
            with rasterio.open(out_fp, 'w', **meta) as dst:
                dst.write(data)


def copy_nonresampled_predictions(in_dir, out_dir, fn_list=None):
    """seperate non-resampled predictions from the evaluation output"""

    fp_list = [os.path.join(in_dir, i) for i in os.listdir(in_dir) if i.endswith('.tif')]
    os.makedirs(out_dir, exist_ok=True)

    if not fn_list:
        # get resolution information from the file names
        arr = [(i, '_'.join(i.split('.tif')[0].split('_')[:-1]), int(i.split('.tif')[0].split('_')[-1])) for i in fp_list]
        # convert arr to dataframe
        df = pd.DataFrame(arr, columns=['fp', 'prefix', 'resolution'])
        # sort df by resolution
        df = df.sort_values('resolution')
        # groupby prefix and select the smallest resolution
        df = df.groupby('prefix').first().reset_index()
        # get the file paths
        fp_list = df['fp'].tolist()
    else:
        # discard fn in fn_list if exist in out_dir
        fn_list = [i for i in fn_list if not os.path.exists(os.path.join(out_dir, i+'.tif'))]
        fp_list = [os.path.join(in_dir, i+'.tif') for i in fn_list if i+'.tif' in os.listdir(in_dir)]
    # copy the files to the output directory while saving the modification date
    list(map(lambda fp: shutil.copy(fp, os.path.join(out_dir, os.path.basename(fp))), tqdm(fp_list)))


def combine_gpks(fp_list, out_fp, layers=None, driver='shp'):
    """Combine geopackages in a directory"""

    # get the list of file paths
    if isinstance(fp_list, str):
        fp_list = [os.path.join(fp_list, i) for i in os.listdir(fp_list) if i.endswith('.gpkg')]

    gdf_list = []

    if layers is None:
        for fp in fp_list:
            gdf = gpd.read_file(fp)
    else:
        for l in layers:
            for fp in fp_list:
                gdf = gpd.read_file(fp, layer=l)
                gdf['layer_name'] = l
                gdf_list.append(gdf)

    # combine list of geopackages
    gdf_out = pd.concat(gdf_list, ignore_index=True)
    gdf_out.drop(columns=[i for i in gdf_out.columns if not i in ['geometry', 'layer_name']], inplace=True)
    gdf_out.to_file(out_fp, driver=driver)


if __name__ == '__main__':
    copy_nonresampled_predictions_ = False
    resample_raster_ = False
    create_tile_index_ = False
    reproject_raster_ = False
    add_crs_transform_ = False

    get_masked_image_path_per_counties_ = False
    get_mean_std_per_country_ = False
    normalize_raster_ = False

    extract_auto_split_ = False
    export_tb_to_csv_ = False
    visualize_res_distribution_ = False

    move_prediction_ = False

    # -----------------------------------------------------------------------
    # Copy non-resampled predictions from the evaluation output
    if copy_nonresampled_predictions_:
        copy_nonresampled_predictions('/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v20240508_data20240508_autoResample/pt_raster_resampled_proj',
                                      '/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v20240508_data20240508_autoResample/pt_raster_proj')

    # -----------------------------------------------------------------------
    # Get masked image path per counties
    if get_masked_image_path_per_counties_:
        get_masked_image_path_per_counties('/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_'
                                           'autoSplit_withFin_res_ha_samplerInfo')

    # -----------------------------------------------------------------------
    # Get mean and std. for each band for each country image sets, need to run extract_mean_std_all.py first
    if get_mean_std_per_country_:
        wd = '/mnt/raid5/DL_TreeHealth_Aerial'
        mean_dict = {}
        std_dict = {}
        dataset_names = [i.split('_global_stats')[0] for i in os.listdir(wd) if '_global_stats' in i]
        for dataset in dataset_names:
            mean_dict.update({dataset: get_mean_std_per_country(dataset, in_dir=wd)[0]})
            std_dict.update({dataset: get_mean_std_per_country(dataset, in_dir=wd)[1]})
        print(mean_dict, std_dict)
        out_mean_dict = {'denmark2020': [0.3543, 0.3946, 0.3799, 0.4767],
                         'finland2021': [0.3661, 0.3777, 0.3671, 0.3235],
                         'germany2022': [0.4281, 0.482, 0.4505, 0.6711],
                         'swiss2022': [0.4323, 0.5509, 0.5452, 0.525],
                         'swiss3b20cm2022': [0.3599, 0.4055, 0.3136, 0.0],
                         'spain2022': [0.4861, 0.4531, 0.402, 0.0],
                         'NAIP2016_2022': [0.4158, 0.4169, 0.3691, 0.498]}
        out_std_dict = {'denmark2020': [0.1323, 0.1121, 0.0986, 0.1293],
                        'finland2021': [0.1684, 0.1604, 0.1474, 0.1396],
                        'germany2022': [0.2006, 0.1738, 0.1549, 0.1651],
                        'swiss2022': [0.2235, 0.2173, 0.1597, 0.2564],
                        'swiss3b20cm2022': [0.1564, 0.1608, 0.1451, 0.0],
                        'spain2022': [0.1605, 0.1402, 0.1373, 0.0],
                        'NAIP2016_2022': [0.1488, 0.1253, 0.1043, 0.1492]}

    # -----------------------------------------------------------------------
    # normalize rasters using mean and std. for each band
    if normalize_raster_:
        out_mean_dict = {'denmark2020': [0.3543, 0.3946, 0.3799, 0.4767],
                         'finland2021': [0.3661, 0.3777, 0.3671, 0.3235],
                         'germany2022': [0.4281, 0.482, 0.4505, 0.6711],
                         'swiss2022': [0.4323, 0.5509, 0.5452, 0.525],
                         'swiss3b20cm2022': [0.3599, 0.4055, 0.3136, 0.0],
                         'spain2022': [0.4861, 0.4531, 0.402, 0.0],
                         'NAIP2016_2022': [0.4158, 0.4169, 0.3691, 0.498]}
        out_std_dict = {'denmark2020': [0.1323, 0.1121, 0.0986, 0.1293],
                        'finland2021': [0.1684, 0.1604, 0.1474, 0.1396],
                        'germany2022': [0.2006, 0.1738, 0.1549, 0.1651],
                        'swiss2022': [0.2235, 0.2173, 0.1597, 0.2564],
                        'swiss3b20cm2022': [0.1564, 0.1608, 0.1451, 0.0],
                        'spain2022': [0.1605, 0.1402, 0.1373, 0.0],
                        'NAIP2016_2022': [0.1488, 0.1253, 0.1043, 0.1492]}

        wd = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin_res_ha_samplerInfo'

        fp_list_txt = {
            # "denmark2020": Path(wd) / "fp_list_DK.txt",
            "finland2021": Path(wd) / "fp_list_FI.txt",
            "germany2022": Path(wd) / "fp_list_DE.txt",
            "swiss2022": Path(wd) / "fp_list_CH.txt",
            # "swiss3b20cm2022": Path(wd) / "fp_list_CH.txt",
            "spain2022": Path(wd) / "fp_list_ES.txt",
            "NAIP2016_2022": Path(wd) / "fp_list_CA.txt"
        }
        for area_name in out_mean_dict.keys():
            if area_name not in ['swiss3b20cm2022', 'denmark2020']:
                normalize_raster(out_dir=Path(wd) / 'normalized_pts',
                                 area_name=area_name,
                                 fp_list_txt=fp_list_txt[area_name],
                                 mean_std_dict={'mean': out_mean_dict, 'std.': out_std_dict})

        fptxt_list = [os.path.join(wd, i) for i in os.listdir(wd) if 'fp_list' in i]
        for fptxt in fptxt_list:
            fp_list = [i.split('\n')[0] for i in open(fptxt, 'r').readlines()]
            fp_list_new = [i.replace('extracted_images_masked', 'normalized_pts') + '\n' for i in fp_list]
            with open(fptxt.replace('fp_list', 'fp_list_normalized'), 'w') as f:
                f.writelines(fp_list_new)


    # -----------------------------------------------------------------------
    # resample harz aerial photos from 20cm to 60cm
    #25 / 60 (spain) # 1 / 3 (germany/denmark)
    if resample_raster_:
        resample_raster(res_factor=10/20,
                        in_dir='/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/pseudo_NIR_random',
                        out_dir='/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/pseudo_NIR_random_20cm')

    # -------------------------------------------
    # Histogram match

    # histogram_match()

    # -------------------------------------------
    # Create tile index shapefile
    if create_tile_index_:
        create_tile_index(wd='./tmp/feature_maps_pca3',
                          out_fp='./tmp/feature_maps_pca3/tile_index.shp',
                          mul_crs=['EPSG:26911', 'EPSG:26910'], out_crs='EPSG:26911')

    # ---------------------------------------------
    # Reproject patch images to be able to use in geoserver...
    if reproject_raster_:
        # reproject_raster(out_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240328v2_256_buffered_'
        #                          'withFin_res_ha/extracted_pts_proj',
        #                  in_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240328v2_256_buffered_wi'
        #                         'thFin_res_ha/extracted_pts')
        reproject_raster(out_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_'
                                 'withFin_res_ha_samplerInfo/normalized_pts_proj',
                         in_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_'
                                'withFin_res_ha_samplerInfo/normalized_pts')

    if add_crs_transform_:
        add_crs_transform(in_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240328v2_256_buffered_withFin_res_'
                                 'ha/normalized_pts',
                          out_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240328v2_256_buffered_withFin_res_'
                                  'ha/normalized_pts_proj',
                          ref_dir ='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240328v2_256_buffered_withFin_'
                                   'res_ha/extracted_pts')

    # -------------------------------------------
    # Extract auto split indices and saved in a new column named auto_split in /qgis/patches/patch_grids.gpkg
    if extract_auto_split_:
        from config.treehealth_5c import TreeHealthSegmentationConfig
        cls = TreeHealthSegmentationConfig()
        cls.__dict__.update({'processed_dir': '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/auto_split',
                             'allow_partial_patches': True, 'allow_empty_patches': True, 'patch_size': 256,
                             'sequential_stride': 256})
        config = init_config(cls)
        a = extract_auto_split(config, save_pdf=True)

    # ---------------------------------------------
    # Export data in tensorboard to a csv file
    if export_tb_to_csv_:
        # export_tb_to_csv(log_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/5c_256_Mar31_04-01-37_11_'
        #                          'Maverickmiaow/tensorboard', scalars='patch_resolutions',
        #                  out_fp='./tmp/patch_resolution.csv')

        export_patch_resolutions(run_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/5c_256_Mar31_04-01-37_11_'
                                 'Maverickmiaow/tensorboard', output_dir='./tmp')

    if visualize_res_distribution_:
        model_version = 'v20240410_08_11'

        in_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/run_5c_256_Apr08_21-46-21_11_Maverickmiaow'
        analyses_patch_res(in_dir)
        df1 = pd.read_csv(os.path.join(in_dir, 'patch_resolutions.csv'))
        in_dir2 = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/run_5c_256_Apr10_22-49-26_8_Maverickmiaow'
        analyses_patch_res(in_dir2)
        df2 = pd.read_csv(os.path.join(in_dir2, 'patch_resolutions.csv'))
        in_dir3 = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/run_5c_256_Apr11_22-43-55_10_Maverickmiaow'
        analyses_patch_res(in_dir3)
        df3 = pd.read_csv(os.path.join(in_dir3, 'patch_resolutions.csv'))

        df = pd.concat([df1, df2, df3])

        list_ = []
        [list_.extend(df[i].values) for i in df.columns[:-4] if i.startswith('Char')]
        df = pd.Series(list_, name="patch_resolutions")
        df = df.to_frame()
        # df['uid'] = list(df)
        data_distribution(data=df, col='patch_resolutions', transform_func=lambda x: x,
                          bins=150, width=1, xlabel='image resolution (m)',
                          out_dir=f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/forEGU2024/{model_version}',
                          prefix='training_nogroups_')

    # ----------------------------------------------
    # Move predictions to vector and raster folder
    if move_prediction_ :
        move_prediction([os.path.join('/mnt/raid5/DL_TreeHealth_Aerial/Merged/predictions/5c_owatershed_v20240330/'
                                      '20240402-195107', i)
                         for i in os.listdir('/mnt/raid5/DL_TreeHealth_Aerial/Merged/predictions/5c_owatersh'
                                             'ed_v20240330/20240402-195107') if i.endswith('.gpkg')],
                        '/mnt/raid5/DL_TreeHealth_Aerial/Merged/predictions/5c_owatersh'
                                             'ed_v20240330/20240402-195107/vectors')


    pass