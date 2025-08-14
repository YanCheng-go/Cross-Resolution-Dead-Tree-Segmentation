import os.path
import typing
from datetime import datetime
from pathlib import Path

from config.base import BaseClassificationConfig, extend_docstring, BaseConfig, BaseSegmentationConfig, \
    BaseRegressionMapConfig
from config.watershed import WatershedConfig


class TreeHealthSegmentationConfig(WatershedConfig):
    """
    Config for the tree health segmentation task

    Parameters
    ----------
    """
    area_name = 'California'

    data_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/CA/training_datasets/20230606/")
    log_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/CA/logs_DeLfoRS")
    n_classes: int = 1
    labeled_areas_file: str = "rectangles_clean.shp"
    target_features_file: str = "segmentations_clean.shp"
    model_type = "unet"
    norm_type_up = "bn"
    norm_type_down = "bn"
    final_skip = True
    loss_function: str = "tversky46_g2"
    backbone = "efficientnet_v2_s" # standard, convnext_small, convnext_tiny, convnext_base
    val_split = 0.1
    test_split = 0.1
    num_workers = 32
    epochs = 2000
    lr = 3e-5
    load = '/mnt/raid5/DL_TreeHealth_Aerial/CA/logs_DeLfoRS/RwandaModel2CaliConfig--epochs-50_Oct04_11-17-17_10_Maverickmiaow/model/BestModel.pth'
    warmup = 10
    val_metric: str = 'loss_count_cumprod/val'
    loss_energy_weight = 1
    loss_sobel_weight = 10
    loss_density_weight = 0.01
    target_col: str = None  #"class"
    in_channels: int = [True, True, True, True]  # if all years a reused this could be: 4 + 4 + 4
    allow_empty_patches: bool = True
    patch_size: int = 256
    sequential_stride: int = 256
    merge_mode: str = 'keep_biggest'  # default keep_first
    batch_size: int = 8
    batch_size_val: int = 256
    dataset_split_by: str = "patches"  # areas or patches
    processed_dir: str = '/mnt/raid5/DL_TreeHealth_Aerial/CA/processed_dir/20230606_affine_debugOverlaps_normalization'
    extract_images_for_areas = False
    save_samples = True
    save_patch_df = True
    save_idf = True
    self_attention = False
    allow_partial_patches = True # try out
    checkpoint_per_epoch = 500
    reference_source: str = "2020"
    project_crs = 'EPSG:26911'
    image_srcs: typing.Dict = {
        "2020": {
            "base_path": Path("/mnt/ssd0/NAIP/intersect_ads2020"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "",
        },
    }

    saved_resampled_patches: bool = False

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseSegmentationConfig, BaseConfig)


class TreeHealthSegmentationPredictionConfig(TreeHealthSegmentationConfig):
    """
    Config for the tree health segmentation task

    Parameters
    ----------
    """

    area_name: str = 'california'

    patch_size: int = 256
    n_edge_removal: int = 10
    sequential_stride: int = 256 - n_edge_removal * 2
    batch_size: int = 128

    # norm_type_down = "default"
    # norm_type_up = "backbone"
    # final_skip = False
    # backbone = "efficientnet_v2_m"
    # self_attention = True

    batch_size_val: int = batch_size
    compression: str = "lzw"
    save_samples = False
    save_patch_df = False
    save_idf = False
    allow_partial_patches = True

    out_prediction_folder = ''


class TreeHealthSegmentationPostprocessingConfig(TreeHealthSegmentationConfig):
    """
    Config for the tree health segmentation task

    Parameters
    ----------
    """
    area_name = 'germany'

    load = '/mnt/raid5/DL_TreeHealth_Aerial/CA/logs_DeLfoRS/TreeHealthSegmentationConfig_Jun06_18-00-49_15_Maverickmiaow/model/BestModel.pth'
    block_size: int = 166 # number of pixels, 166 -> quantify bias of count per ha given that the spatial resolution of NAIP is 0.6 meter
    bias_correction_mode: str = 'static'
    batch_size: int = 1
    num_workers = 32
    # norm_type_down = "default"
    # norm_type_up = "backbone"
    # final_skip = False
    # backbone = "efficientnet_v2_m"

    if area_name == 'california':
        image_srcs: typing.Dict = {
            "2020": {
                "base_path": Path("/mnt/ssd0/NAIP/intersect_ads2020"),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
            },
        }
    elif area_name == 'germany':
        project_crs = 'EPSG:4326'
        patch_size = 128
        site = 'germany'
        res = '' #'_60cm'
        img_year = 2022
        data_date = '20240130'
        load = '/mnt/raid5/DL_TreeHealth_Aerial/Germany/logs_DeLfoRS/patch-size-128_Feb05_16-18-34_12_Maverickmiaow/model/BestModel.pth'
        block_size: int = 500  # number of pixels, 166 -> quantify bias of count per ha given that the spatial resolution of NAIP is 0.6 meter

        data_dir: str = Path(f"/mnt/raid5/DL_TreeHealth_Aerial/Germany/training_dataset/{data_date}")
        log_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/Germany/logs_DeLfoRS")
        labeled_areas_file: str = "labeled_areas.shp"
        target_features_file: str = "target_features.shp"
        processed_dir: str = f'/mnt/raid5/DL_TreeHealth_Aerial/Germany/processed_dir/20240130_{patch_size}{res}'

        if res == '':
            filelist_path_name = f'fp_list_{img_year}_org.txt'
        elif res == '_60cm':
            filelist_path_name = f'fp_list_{img_year}.txt'

        reference_source = f'{site}_{img_year}{res}'
        image_srcs: typing.Dict = {
            f"{site}_{img_year}{res}": {
                "base_path": Path(f"/mnt/raid5/DL_TreeHealth_Aerial/Germany/Ortho{img_year}_{site}{res}"),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
                "filelist_path": f'/mnt/raid5/DL_TreeHealth_Aerial/Germany/{filelist_path_name}'
            },
        }


class RwandaModel2CaliConfig(BaseRegressionMapConfig):
    area_name = 'california'

    data_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/CA/training_datasets/20230606/")
    log_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/CA/logs_DeLfoRS")
    n_classes: int = 1
    model_type = "unet"
    norm_type_up = "bn"
    norm_type_down = "bn"
    loss_function: str = "kd_binary"
    backbone = "efficientnet_v2_s"  # standard, convnext_small, convnext_tiny, convnext_base
    val_split = 0.1
    test_split = 0.1
    num_workers = 12
    epochs = 7000
    in_channels: int = 4  # if all years a reused this could be: 4 + 4 + 4
    patch_size: int = 256
    sequential_stride: int = 256
    batch_size: int = 8
    batch_size_val: int = 256
    dataset_split_by: str = "patches"  # areas or patches
    processed_dir: str = '/mnt/raid5/DL_TreeHealth_Aerial/CA/processed_dir/20230606_affine_debugOverlaps'
    save_samples = True
    save_patch_df = True
    save_idf = True
    self_attention = False
    final_skip = True
    allow_partial_patches = True
    checkpoint_per_epoch = 500
    run_dir_suffix = ''

    project_crs = 'EPSG:4326'
    reference_source = "2020"
    target_image_source = "RwandaModel"

    image_srcs: typing.Dict = {
        "2020": {
            "base_path": Path("/mnt/ssd0/NAIP/intersect_ads2020"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "",
        },
        "RwandaModel": {
            "base_path": Path("/mnt/raid5/DL_TreeHealth_Aerial/CA/output_predictions_alivetrees/intersect_ads2020/rasters_compress"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "",
        },
    }