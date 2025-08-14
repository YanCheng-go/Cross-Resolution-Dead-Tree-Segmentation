import json
import os
from pathlib import Path

from postprocessing.ordinal_watershed_evaluation import OrdinalWatershedPostprocessor

from config.treehealth_5c import TreeHealthSegmentationPostprocessingConfig, TreeHealthSegmentationConfig
from train.base import init_config


if __name__ == '__main__':

    config_pred = init_config(TreeHealthSegmentationPostprocessingConfig())
    config_path = os.path.join(os.path.dirname(os.path.dirname(Path(config_pred.load))), 'config.json')
    with open(config_path) as f:
        config_train = json.loads(f.read())

    if config_pred.config_file is not None and config_pred.config_file != '' and os.path.exists(config_pred.config_file):
        with open(config_pred.config_file) as f:
            config_ = json.loads(f.read())
            config_pred.__dict__.update(config_)
            cls = config_pred

    else:
        cls = init_config(TreeHealthSegmentationConfig())
        updated = dict([(i, config_train.get(i)) for i in cls.__dict__ if cls.__dict__.get(i)
                        != config_train.get(i) and i != '__doc__'])
        cls = TreeHealthSegmentationPostprocessingConfig()
        cls.__dict__.update(updated)

    # Add configs from the cml
    config = init_config(cls)

    # Make sure the configuration for normalization is the same as the model loaded
    update_normalization = dict([(i, config_train.get(i)) for i in ['normalize', 'normalize_by_dataset', 'normalize_by_imagenet', 'in_channels', 'model_type', 'band_sequence']])

    # set as false if none
    if update_normalization['normalize'] is None:
        update_normalization['normalize'] = False
    if update_normalization['normalize_by_dataset'] is None:
        update_normalization['normalize_by_dataset'] = False
    if update_normalization['normalize_by_imagenet'] is None:
        update_normalization['normalize_by_imagenet'] = False

    if any([i is None for i in update_normalization.values()]):
        raise ValueError("Normalization parameters are not set correctly.")
    config.__dict__.update(update_normalization)
    config.__dict__.update({'pretrained_ckpt': None})
    config.__dict__.update({'reset_head': False})

    postprocessor = OrdinalWatershedPostprocessor(config)
    os.makedirs(config.report_folder, exist_ok=True)
    postprocessor.calculate_count_bias(save_df=True, proj_crs='EPSG:4326')