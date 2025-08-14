import typing
from pathlib import Path

from config.base import BaseSegmentationConfig, extend_docstring, BaseConfig


class WatershedConfig(BaseSegmentationConfig):
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """

    # Class number is index in this list
    # Therefore, the order matters here. bg must be the first class and u must be the last. u is ignored
    target_classes: typing.List[str] = None
    allow_partial_patches: bool = False
    rasterize_all_touched = True
    pipeline_test: bool = False

    # Model related config
    model_type: str = "unet"
    in_channels: int = 3
    # currently only support for binary class segmentation
    n_classes: int = 1  # (internally more (e.g. for energy, sobelx, sobely, density))
    # maximal distance to consider
    # (0=always background, 1=border, 2=2nd energy level, max level depends on number of bins)
    n_energy_bins: int = 10
    loss_function = "bce"
    loss_function_sobel = "smoothl1"
    loss_function_density = "smoothl1"
    loss_function_energy = "focal"
    loss_energy_weight = 1
    loss_sobel_weight = 0.1
    loss_density_weight = 0.025
    ordinal_connect = True
    scaler_weights = None
    fuse_mi = 3
    fuse_mx = 11
    fuse_interval = 2
    fuse_in = False
    fuse_out = False
    rescale = False
    auto_resample: bool = False  # for evaluation and test??? apply automatic resampling or not...
    max_value: typing.Union[int, None] = 255.
    report_folder = Path("./reports")
    normalize: bool = False
    save_out_rasters: bool = False

    apply_count_weights: bool = False
    apply_edge_weights: bool = False
    edge_weight = 10
    edge_threshold_pixel = 3
    edge_threshold_percent = 0.2
    small_object_threshold = None

    nir_drop_chance: float = 0.0
    normalize_by_imagenet: bool = False
    pretrained_ckpt: str = None  # only for unet based pretrained models...
    run_dir_suffix = 'run'
    evaluate_datasets: str = 'val,test'
    config_file: str = None

    saved_resampled_patches: bool = False

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseSegmentationConfig, BaseConfig)


class WatershedRwandaConfig(BaseSegmentationConfig):
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """

    # Dataset related config
    data_dir = Path('/home/ankit/Vision/Rwanda/TrainingDataV1')
    project_crs: str = 'EPSG:4326'
    labeled_areas_file: str = 'labeled_rectangles.gpkg'
    target_features_file: str = 'tree_labels.gpkg'
    target_col: str = None
    split_col: str = 'dataset_split'

    # Class number is index in this list
    # Therefore, the order matters here. bg must be the first class and u must be the last. u is ignored
    target_classes: typing.List[str] = None
    patch_size: int = 384
    sequential_stride: int = 384
    allow_partial_patches: bool = False
    rasterize_all_touched = True
    pipeline_test: bool = False

    # Model related config
    model_type: str = "unet_segmentation_models_pytorch"
    lr: float = 3e-4
    val_split: float = 0.2
    test_split: float = 0.2
    lr_scheduler: str = None  # "cosinewr"
    lr_scheduler_warmup: int = 2
    backbone_warmup: int = 1
    in_channels: int = 3
    n_classes: int = 4  # Segmentation mask, energey, sobelx, sobely

    # Cache config
    processed_dir: str = Path("./watershed_rwanda_cache")
    save_idf: bool = True
    save_labeled_areas_df: bool = True
    save_samples: bool = False
    save_patch_df: bool = True
    extract_images_for_areas: bool = True

    # Image src
    reference_source = "aerial_20"
    image_srcs: dict = {
        'aerial_20': {
            'base_path': Path('/home/ankit/Vision/Rwanda/Final_2008_VHR_Images'),
            'image_file_type': '.tif',
            'image_file_prefix': '',
            "image_file_postfix": "",
        }
    }

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseSegmentationConfig, BaseConfig)


class Hen_WatershedRwandaConfig(WatershedRwandaConfig):
    """ Configuration of all parameters used in preprocessing.py, training.py and prediction.py """

    # Dataset related config
    data_dir = Path('/home/qmk959/data/Rwanda/TrainingDataV1')

    save_samples: bool = True
    extract_images_for_areas: bool = False

    image_srcs: dict = {
        'aerial_20': {
            'base_path': Path('/home/qmk959/data/Rwanda/Final_2008_VHR_Images'),
            'image_file_type': '.tif',
            'image_file_prefix': '',
            "image_file_postfix": "",
        }
    }

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, WatershedRwandaConfig, BaseSegmentationConfig, BaseConfig)
