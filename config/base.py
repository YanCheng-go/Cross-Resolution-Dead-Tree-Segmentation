import pathlib
import typing
from pathlib import Path

import pyproj


class BaseConfig:
    """
    Basic config for all the tasks, such as classification, segmentation and regression in this repository

    Parameters
    ----------
    job_name: str
        What you want to call your task.
    seed: int
        Random seed for reproducibility.
    data_dir: str
        The path to directory containing the data
    target_features_file: str
        A shapefile/geopackage containing the labeled polygons of target class. Eg., trees or water bodies. Set to None, when predicting.
    target_col: str
        In the target_features_file, the column containing information about the target features. Eg., tree species. If None, then it is treated as a binary problem, otherwise as a multiclass problem
    reference_source: str or None
        The reference image source. When the value is None, use the keys in image_srcs as reference_source
    project_crs: str
        CRS most spatial operations are performed in.
    image_srcs: typing.Dict
        Dictionary describing all images sources and their paths; Can NOT be overridden from the command line
    patch_size: int
        Height and width of the patches
    merge_mode: str
        How to merge overlapping images from same source (either 'keep_first' or 'keep_last').
    log_dir:  str
        Directory to save models and tensorboard output
    log_stdout: bool
        Whether to log to the stdout in addition to the logging file
    processed_dir: str
        Directory to save processed samples and labeled areas dataframes. Is not used if  "None ".
        Works only in conjunction with 'save_samples' or 'save_patch_df'.
    save_samples: bool
        If processed samples should be saved in the 'processed_dir'.
        Works only if 'processed_dir' is set.
    save_patch_df: bool
        If processed labeled areas should be saved in the 'processed_dir'.
        Works only if 'processed_dir' is set.
    save_idf: bool
        If the image table should be saved in the 'processed_dir'.
        Works only if 'processed_dir' is set.
    split_col: str
        Column in area frame that indicates split for training, validation, and testing
    epochs: int
        Maximum number of training epochs
    batch_size: int
        Batch size for training
    batch_size_val: int
        Batch size of validation
    n_grad_accumulation: int
        Number of batches over which the gradients are accumulated i.e. affectively increases the batch size.
    pipeline_test: bool
        Indicates if script is only testing the pipeline and actually running fully
    lr: float
        Learning rate
    opt_eps: float
        Epsilon parameter of Adam
    load: str
        Path to load model from file, filetype: .pth
    reset_head: bool
        If true, resets the final layer (good for using pre-trained networks)
    val_split: float
        Percent of the data that is used as validation [0.0-1.0]
    test_split: float
        Percent of the data that is used as test set [0.0-1.0]
    val_metric_opt: str
        Optimization used for given val_metric (either "min" or "max"). (default: "min")
    num_workers: int
        How many data workers per dataloader
    multi_devices:  typing.List
        ids of gpus that should be used (e.g., multi_devices 0 1)
    model_type: str
        Type of model to train
    in_channels: int
        Number of input channels that are passed to the model
    n_classes: int
        Number of output classes predicted by the model
    loss_function: str
        Loss function to use for training the model
    use_augmentation: bool
        Whether to use augmentation
    norm_type_up: str
        Normalization of up path, e.g., bn
    norm_type_down: str
        Normalization of down path, e.g., bn
    activation: str
        Activation function to use, e.g., elu
    backbone: str
        Architecture of backbone_model, e.g.,
        "standard ",  "mnasnet ",  "resnet18 ",  "resnet50 ",  "swin_s",  "swin_b",
        "efficientnet_b0" -- "efficientnet_b7", "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l",
        "convnext_tiny", "convnext_small", "convnext_base", "convnext_large",
    initial_stride: [int, "default", None]
        Initial stride of backbone layer (if downsampling should occur in first layer). e.g., "default" or int.
        Default: 1
    model_version: str
        The model version to be used (if instantiated)
    verbose_logging: bool
        Whether to use verbose_logging.
    checkpoint_per_epoch: int
        After how epoch a checkpoint is written
    lr_scheduler: str
        Either None,  "linear ", or  "cosinewr "
    lr_scheduler_warmup: int
        After how many epochs to start using the lr_scheduler
    backbone_warmup: int
        Number of warmup epochs for backbone network
    warmup: int
        Number of warmup epochs for entire network except for the final layer
    self_attention: bool
        Whether to use self attention
    blur: bool
        Whether to use blur augmentation in the unet upsampling
    blur_final: bool
        Whether to use blur augmentation in the final unet upsampling block (only works if blur is also set)
    final_skip: bool
        Whether to concatentate first feature map to last output layer
    upsampling: str
        Upsampling in decoder, e.g., pixelshuffle, nearest, bilinear
    pyramid_level: int
        resample scales, always positive values, only upsampling
    append_data: bool
        append data in image_srcs to reference_source or save independently.
    year_col: str
        the column in targe_features.shp that indicates the year of images used to create the target features.
    band_sequence: list
        the list of band index in the order of R, G, B, NIR...
    weighted_sampling: bool
        apply randomweightedsampler or not, if yes, need to define the sampler...
    """
    # ---------------------------
    # ------- DATA CONFIG -------
    # ---------------------------
    job_name: str = "nn_train"
    seed: int = 42
    data_dir: str = Path("./data/tree_species")
    target_features_file: str = "shapefile/tree_species_test.gpkg"
    target_col: str = None
    target_classes: typing.List[str] = None
    reference_source: str = "planet_5m_12y"
    image_srcs: typing.Dict = {
        "planet_5m_12y": {
            "base_path": Path("./data/tree_species/planet_5m_12y"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "",
            "recursive": False
        },
    }
    patch_size: int = 512
    allow_partial_patches: bool = False
    append_data: bool = True
    year_col: str = None
    band_sequence: typing.Union[typing.Literal["default", None], typing.List[int]] = None
    weighted_sampling: bool = False
    out_prediction_folder: str = "predictions"

    # ---------------------------
    # ----- TRAINING CONFIG -----
    # ---------------------------
    log_dir: str = "./runs"
    log_stdout: bool = True
    processed_dir: str = None
    merge_mode: str = "keep_first"
    save_samples: bool = False
    save_labeled_areas_df = False
    save_patch_df: bool = False
    save_idf: bool = False
    epochs: int = 500
    batch_size: int = 8
    batch_size_val: int = batch_size
    n_grad_accumulation: int = 1
    lr: float = 3e-4
    opt_eps: float = 1e-8
    load: str = None
    reset_head: bool = False
    val_split: float = 0.2
    test_split: float = 0.2
    val_metric: str = "loss/val"
    val_metric_opt: str = "min"
    num_workers: int = 0
    multi_devices: typing.List[str] = None
    model_type: str = "unet"
    loss_function: str = "bce"
    in_channels: typing.Union[int, typing.List[bool]] = 3
    n_classes: int = 1
    use_augmentation: bool = True
    norm_type_up: str = None
    norm_type_down: str = "default"
    activation: str = "default"
    backbone: str = "standard"
    initial_stride: typing.Union[int, typing.Literal["default", None]] = 1
    model_version: str = "v0"
    verbose_logging: bool = True
    checkpoint_per_epoch: int = 100
    project_crs: str = "EPSG:4326"
    lr_scheduler: str = None
    lr_scheduler_warmup: int = 0
    backbone_warmup: int = 0
    warmup: int = 0
    self_attention: bool = False
    final_skip: bool = False
    blur: bool = False
    blur_final: bool = False
    upsampling: str = "nearest"
    split_col: str = None
    rasterize_all_touched: bool = False
    pipeline_test: bool = False
    pyramid_level: int = 1
    iou_thresh: float = 0.5
    device:  typing.Union[str, typing.Literal["default", None]] = "cuda"

    saved_resampled_patches = False

    def get(self, name, default=None):
        return getattr(self, name, default)

    def _serialize_dict(self, d):
        s = {}
        for k, v in d.items():
            if issubclass(type(v), (pathlib.PosixPath, pathlib.WindowsPath)):
                s[k] = str(v)
            elif isinstance(v, dict):
                s[k] = self._serialize_dict(v)
            else:
                s[k] = v
        return s

    def _dump(self):
        d = {}
        # https://stackoverflow.com/questions/980249/difference-between-dir-and-vars-keys-in-python
        for k in dir(self):
            if not k.startswith('_') and not callable(getattr(self, k)):
                v = getattr(self, k)
                if issubclass(type(v), (
                        pathlib.PosixPath, pathlib.WindowsPath,
                        pyproj.CRS)):  # A path object, which is not JSON serializable
                    d[k] = str(v)
                elif isinstance(v, dict):
                    d[k] = self._serialize_dict(v)
                else:
                    d[k] = v
        return d


class BaseSegmentationConfig(BaseConfig):
    """
    Config for a basic segmentation task, adds to the BaseConfig

    Parameters
    ----------
    labeled_areas_file: str
        The file of the shapefile/geopackage containing the areas where are target features are labeled
    extract_images_for_areas: bool
        Extract relevant parts of the images (i.e. parts that overlap with area) into separate files.
    initial_stride: int
        Stride of the first layer in the model
    dataset_split_by: str
        Either "patches" or "areas"
    n_edge_removal: int
        [prediction parameter] How many pixel on each side should be discarded to avoid edge artifacts
    """
    extract_images_for_areas: bool = False
    initial_stride: typing.Union[int, typing.Literal[
        "default", None]] = 1  # often the better choice for unet-type architectures
    labeled_areas_file: str = 'reprojected_areas.shp'
    patch_size: int = 512
    sequential_stride: int = 384
    allow_empty_patches: bool = True
    dataset_split_by: str = "areas"  # areas or patches
    n_edge_removal: int = 0

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseConfig)


class BaseRegressionMapConfig(BaseConfig):
    """
    Config for a basic regression man task, adds to the BaseConfig

    Parameters
    ----------
    labeled_areas_file: str
        The file of the shapefile/geopackage containing the areas where are target features are labeled.
        If set to "auto", the intersections of target_image_source and refrence_image_source are used.
    """
    labeled_areas_file: str = 'auto'
    target_image_source: str = "y"
    loss_function: str = "smooth_l1"
    model_type: str = "unet"

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseConfig)


class SenegalHeightCorrectionConfig(BaseRegressionMapConfig):
    in_channels: int = 2
    n_classes: int = 1
    loss_function: str = "mse"
    data_dir: str = Path("./data/canopy_senegal/")
    area_file: str = 'shapes/boundaries.shp'
    reference_source: str = "dasm"
    target_image_source: str = "dasm"

    image_srcs: typing.Dict = {
        "dasm": {
            "base_path": Path("./data/canopy_senegal/heights"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "dasm",
            "recursive": False,
        }
    }


class SenegalHeightConfig(BaseRegressionMapConfig):
    """
    Config for Senegal height prediction task

    Parameters
    ----------

    """
    in_channels: int = 6
    n_classes: int = 1
    data_dir: str = Path("./data/canopy_senegal/")
    labeled_areas_file: str = 'shapes/boundaries.shp'
    reference_source: str = "nasa"
    target_image_source: str = "dasm"

    image_srcs: typing.Dict = {
        "nasa": {
            "base_path": Path("./data/canopy_senegal/"),
            "image_file_type": ".jp2",
            "image_file_prefix": "",
            "recursive": False,
            "image_file_postfix": "",
        },
        "dasm": {
            "base_path": Path("./data/canopy_senegal/heights"),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "recursive": False,
            "image_file_postfix": "dasm",
        }
    }

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseRegressionMapConfig, BaseConfig)


class BaseClassificationConfig(BaseConfig):
    """
    Config for a basic classification task

    Parameters
    ----------
    metrics: "list[str]"
        Which metrics to use for validation/test
    out_poly: str
        Where to save the predictions
    target_col: str
        Names of the target column (class with indicator 1)
    max_area: float
        Maximum size of a single polygon
    final_factor: [int, "default", None]
        If the output of the backbone should be increased before classification
        (default: "default", take backbone settings)
    n_final_mlp: int
        Number of mlp layers after backbone before classification
        (default: 0)
    final_dropout: [float, "default", None]
        Probability to apply dropout before classification.
        (default: "default", take backbone settings)
    is_fully_convolutional: bool
        If model should be fully convolutional (could be transfered to segmentation)..
        (default: False)
    add_target_mask_to_input: bool
        If target mask (when available) should be added to the input. (default: True)
    """
    model_type: str = "classifiernorm"
    norm_type_in: str = "bn"
    labeled_areas_file: str = None
    in_channels: int = 3
    n_classes: int = 1
    metrics: typing.List[str] = ["accuracy", "f1", "ap", "precision", "recall"]
    out_poly: str = "predictions"
    target_col: str = "y"
    backbone: str = "mnasnet"
    max_area: float = 0
    patch_size: int = 128
    final_factor: typing.Union[int, typing.Literal["default"], None] = "default"
    n_final_mlp: int = 0
    final_dropout: typing.Union[float, typing.Literal["default"], None] = "default"
    is_fully_convolutional: bool = False
    add_target_mask_to_input: bool = True  # adds extra input channel

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseConfig)


class BaseRegressionConfig(BaseConfig):
    """
    Config for a basic Regression task

    Parameters
    ----------
    out_poly: str
        Where to save the predictions
    target_cols: list[str]
        List of the target column names
    max_area: float
        Maximum size of a single polygon
    final_factor: [int, "default", None]
        If the output of the backbone should be increased before final layer
        (default: "default", take backbone settings)
    n_final_mlp: int
        Number of mlp layers after backbone before final layer
        (default: 0)
    final_dropout: [float, "default", None]
        Probability to apply dropout before final layer.
        (default: "default", take backbone settings)
    is_fully_convolutional: bool
        If model should be fully convolutional (could be transferred to segmentation)..
        (default: False)
    add_target_mask_to_input: bool
        If target mask (when available) should be added to the input. (default: True)
    """
    model_type: str = "classifiernorm"
    norm_type_in: str = "bn"
    labeled_areas_file: str = None
    in_channels: int = 3
    n_classes: int = 1
    out_poly: str = "predictions"
    target_col: str = ["y"]
    backbone: str = "efficientnet_v2_s"
    max_area: float = 0
    loss_function: str = "mse"
    patch_size: int = 128
    final_factor: typing.Union[int, typing.Literal["default"], None] = "default"
    n_final_mlp: int = 0
    final_dropout: typing.Union[float, typing.Literal["default"], None] = "default"
    is_fully_convolutional: bool = False
    add_target_mask_to_input: bool = False  # adds extra input channel

    def __init__(self) -> None:
        self.__doc__ = extend_docstring(self, BaseConfig)


def extend_docstring(*args):
    import docstring_parser
    docs = []
    for ar in args:
        docs.append(docstring_parser.parse(ar.__doc__, style=docstring_parser.common.DocstringStyle.NUMPYDOC))
    r = docs[0]
    for i in docs[1:]:
        r.meta = i.meta + r.meta
    return docstring_parser.compose(r)


def config_data_param_compare(conf1, conf2):
    if type(conf1) != dict:
        conf1 = conf1._dump()
    if type(conf2) != dict:
        conf2 = conf2._dump()

    kys = ["data_dir", "target_features_file", "reference_source", "image_srcs", "patch_size", "sequential_stride",
           "processed_dir", "save_samples", "save_labeled_areas_df", "save_patch_df", "save_idf",
           "extract_images_for_areas"]

    sm = True
    ns = []
    for k in kys:
        if k in conf1 and conf1[k] != conf2[k]:
            sm = False
            ns.append(k)
    return sm, ns
