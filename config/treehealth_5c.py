import os
import typing
from datetime import datetime
from pathlib import Path

from config.treehealth import TreeHealthSegmentationConfig


class TreeHealthSegmentationConfig(TreeHealthSegmentationConfig):
    # runtime parameters
    area_name = '5c'  # CH4b25_ES3b25_CA4b60_DE4b20
    train_date = datetime.now().strftime("%Y%m%d")
    data_date = '20240430'  #'CA20231011-DE20240206-CH20240319-ES20240325-FI20220824'
    imgaug_date = '20240430'

    # model parameters
    load: str = None
    checkpoint_per_epoch = 20
    epochs = 200

    apply_count_weights: bool = False
    apply_edge_weights: bool = False

    # Image augmentation related parameters
    auto_resample = True
    weighted_sampling = True
    normalize = True  # no normalization during the training, but the dataset could have been pre-normalized
    max_value = 255. if normalize else 0  # to convert data from 0 - 1 before normalization
    rescale = True
    use_augmentation = True

    batch_size: int = 8
    batch_size_val: int = 16 if auto_resample else 128
    patch_size = 256
    sequential_stride: int = int(patch_size)
    val_metric: str = "loss/val"
    model_type: str = "unet"
    self_attention: bool = False
    # watershed parameters
    n_energy_bins: int = 5
    if epochs == 0:
        run_dir_suffix = f'test_{area_name}_{patch_size}'
    elif epochs == 1:
        run_dir_suffix = f"one_epoch_{area_name}_{patch_size}_test"
    else:
        run_dir_suffix = f"run_{area_name}_{patch_size}"

    # labeled areas and features
    split_col = 'dataset_sp' #'auto_split' or user defined split column name, e.g., 'split_sp'
    dataset_split_by = 'areas'  # 'areas' or 'patches'
    allow_empty_patches: bool = True
    if split_col is None and dataset_split_by == "patches":
        allow_partial_patches: bool = True
        data_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/5c_20240405_autoSplit")
        labeled_areas_file: str = "labeled_areas.shp"  # need to have uid field if using weightedsampler using spatial clusters
        target_features_file: str = "target_features.shp"
    if split_col is not None and dataset_split_by == "areas":
        allow_partial_patches: bool = False
        data_dir: str = Path(f"/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/{area_name}_{data_date}")
        labeled_areas_file: str = "labeled_areas_buffered.shp"
        target_features_file: str = "target_features.shp"

    # image sources
    normalize_by_dataset = False  # if True, normalize in training, if False, use pre-processed normalized images
    extract_images_for_areas = False
    append_data = False
    reference_source = 'germany20cm_2022'  # Just a name in this case
    # for user defined split, with or without pre-processed region-specific normalization of the images
    if dataset_split_by == "areas" and split_col is not None:
        processed_dir = (f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/{imgaug_date}_{data_date}_{patch_size}'
                         f'_noPart_{area_name}')
        save_samples = True
        save_patch_df = True
        save_idf = True

        norm = ''
        band_sequence = [0, 1, 2, 3]

        # image_srcs_list = ['germany20cm_2022', 'swiss25cm_2022'] #['spain25cm_2022', 'germany20cm_2022', 'california60cm_2020', 'swiss25cm_2022', 'spain25cm_2022', 'finland40cm_2021']
        image_srcs: typing.Dict = {
            "germany20cm_2022": {
                "base_path": Path(f"/mnt/raid5/DL_TreeHealth_Aerial/Germany/Ortho2022_germany"),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
                "filelist_path": f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin'
                                 f'_res_ha_samplerInfo/fp_list_{norm}DE.txt'
            },

            "california60cm_2020": {
                "base_path": Path("/mnt/ssd0/NAIP/intersect_ads2020"),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix":"",
                "filelist_path": "/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin_"
                                 f"res_ha_samplerInfo/fp_list_{norm}CA.txt"
            },

            "swiss20cm_2022": {
                "base_path": Path('/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/rgbi_norm_clip'),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
                "filelist_path": "/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin_res_"
                                 f"ha_samplerInfo/fp_list_{norm}CH.txt"
            },

            # "spain25cm_2022": {
            #     "base_path": Path('/mnt/raid5/DL_TreeHealth_Aerial/Spain_additional/pseudo_NIR_zeros'),
            #     "image_file_type": ".tif",
            #     "image_file_prefix": "",
            #     "image_file_postfix": "",
            #     "filelist_path": "/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin_res"
            #                      f"_ha_samplerInfo/fp_list_{norm}ES.txt"
            # },

            "finland40cm_2021": {
                "base_path": Path('/mnt/raid5/DL_TreeHealth_Aerial/Finland/images/vnir_epsg32635_40cm'),
                "image_file_type": ".jp2",
                "image_file_prefix": "",
                "image_file_postfix": "",
                "filelist_path": f"/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin_"
                                 f"res_ha_samplerInfo/fp_list_{norm}FI.txt"
            },
        }
        # image_srcs = {key: image_srcs.get(key) for key in image_srcs_list}

    # For auto split... used real labled areas (merged) for real mask, and need to allow partial if there are ares with labels smaller than patch size...,
    # could extract images and use that for the image source for manually determined sets...-> auto splite -> allow partial -> split patch grids -> buffer -> do not allow partial + extracted images as input image
    if (dataset_split_by == "patches" and split_col is None):
        processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/20240405_256_autoSplit_withFin_res_ha'
        save_samples = False
        save_patch_df = True
        save_idf = True

        band_sequence = [0, 1, 2, 3]
        image_srcs: typing.Dict = {
            "germany20cm_2022": {
                "base_path": Path(f"/mnt/raid5/DL_TreeHealth_Aerial/Germany/Ortho2022_germany"),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
                "filelist_path": f'/mnt/raid5/DL_TreeHealth_Aerial/Germany/fp_list_2022_org.txt'
            },

            "california60cm_2020": {
                "base_path": Path("/mnt/ssd0/NAIP/intersect_ads2020"),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
            },

            "swiss25cm_2022": {
                "base_path": Path('/mnt/raid5/DL_TreeHealth_Aerial/Switzerland/rgbi_norm_clip'),
                "image_file_type": ".tif",
                "image_file_prefix": "",
                "image_file_postfix": "",
            },

            # "spain25cm_2022": {
            #     "base_path": Path('/mnt/raid5/DL_TreeHealth_Aerial/Spain_additional/pseudo_NIR_zeros'),
            #     "image_file_type": ".tif",
            #     "image_file_prefix": "",
            #     "image_file_postfix": "",
            #     "filelist_path": "/mnt/raid5/DL_TreeHealth_Aerial/Spain_additional/fplist_pseudo_NIR_all.txt"
            # },

            "finland40cm_2021": {
                "base_path": Path('/mnt/raid5/DL_TreeHealth_Aerial/Finland/images/vnir_epsg32635_40cm'),
                "image_file_type": ".jp2",
                "image_file_prefix": "",
                "image_file_postfix": "",
                #"filelist_path": ""
            },
        }

    # semi-static parameters
    fuse_interval = 3
    fuse_in = False
    fuse_out = False
    loss_function_sobel = "smoothl1"
    loss_function_density = "smoothl1"
    loss_function_energy = "focal"
    loss_energy_weight = 1
    loss_sobel_weight = 0.01
    loss_density_weight = 0.025
    ordinal_connect = True
    loss_function: str = "tversky46_g2"
    backbone = "efficientnet_v2_s"  # standard, convnext_small, convnext_tiny, convnext_base
    lr: float = 3e-4
    lr_scheduler: str = 'multi_steps'
    reset_head: bool = False
    val_split: float = 0.1
    test_split: float = 0.1
    val_metric_opt: str = "min"
    n_classes: int = 1
    in_channels: typing.Union[int, typing.List[bool]] = [True, True, True, True]
    year_col: str = None
    project_crs = 'EPSG:4326'
    merge_mode: str = "keep_biggest"
    log_dir: str = Path("/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS")
    warmup = 0
    lr_scheduler_warmup = 1
    device = 'cuda'
    num_workers = 16
    saved_resampled_patches: bool = False


class TreeHealthSegmentationPredictionConfig(TreeHealthSegmentationConfig):
    area_name: str = 'spain25cm_2022'
    train_date = '20240503'

    # not needed for hpc
    out_prediction_folder: str = "/mnt/raid5/DL_TreeHealth_Aerial/Merged/predictions/{}_owatershed_v{}/{}".format(
        area_name, train_date, datetime.now().strftime("%Y%m%d-%H%M%S"))

    band_sequence = [0, 1, 2, 3]
    reference_source = TreeHealthSegmentationConfig.reference_source
    image_srcs: typing.Dict = {
        reference_source: {
            "base_path": Path("/mnt/raid5/DL_TreeHealth_Aerial/ortho_labels/resampled_10cm"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "",
            "filelist_path": "/mnt/raid5/DL_TreeHealth_Aerial/ortho_labels/prediction_list_20241022_test.txt"
            #'./test_skysat.txt'
            # f'./fp_list_test.txt'  # Issues with the geometry...
            # "filelist_path": "/mnt/raid5/DL_TreeHealth_Aerial/Merged/predictions/denmark/aarhus_test_sites.txt"
        },
    }

    patch_size = 256
    n_edge_removal: int = 10
    sequential_stride: int = patch_size - n_edge_removal * 2
    batch_size_val: int = 32
    compression: str = "lzw"
    save_samples = True
    save_patch_df = True
    save_idf = True
    allow_partial_patches = True
    allow_empty_patches = True
    # processed_dir = None
    reset_head = False


class TreeHealthSegmentationPostprocessingConfig(TreeHealthSegmentationConfig):
    run_dir_suffix = "post_evaluation_data"

    area_name = '5c'
    train_date = '20240411'
    data_date = '20240411'
    imgaug_date = '20240412'
    patch_size = 256

    processed_dir = (f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/{imgaug_date}_{data_date}_{patch_size}'
                     f'_noPart_{area_name}')

    report_folder = f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v{train_date}'
    load = ('/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/run_5c_256_Apr11_22-43-55_10'
            '_Maverickmiaow/model/BestModel.pth')

    evaluate_datasets = 'train,val,test'

    # image augmentation related parameters
    auto_resample = True  # auto resample every patch
    use_augmentation = True
    normalize = True
    normalize_by_dataset = False
    max_value = 255. if normalize else 0
    assert ((auto_resample is True and use_augmentation is True) or
            (auto_resample is False and use_augmentation is False)), \
        "auto_resample and use_augmentation must be both True or False"
    rescale = True  # save scale_factors to the patch dataframe
    weighted_sampling = True  # save weight_vars to the patch dataframe
    apply_count_weights = False  # when generating processed data here, set as True
    apply_edge_weights = False  # when generating processed data here, set as True
    save_out_rasters = True  # save predictions or not
    if save_out_rasters:
        patch_size = 256
        n_edge_removal: int = 10
        sequential_stride: int = patch_size - n_edge_removal * 2

    # model related parameters
    batch_size_val = 16 if auto_resample else 128

    # Dataset related config
    extract_images_for_areas = False
    allow_partial_patches = False
    allow_empty_patches = True

    # semi-static parameters
    epochs = 1
    num_workers = 16
    save_samples = True
    save_patch_df = True
    save_idf = True
    reset_head = False

    bias_correction_mode = 'static'
    block_size: int = 166  # number of pixels, 166 -> quantify bias of count per ha given that the spatial resolution of NAIP is 0.6 meter

    device = 'cpu'
    medium_area_range_s = 35
    medium_area_range_m = 389
    iou_thresh = 0.50