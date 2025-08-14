import json
import os
from pathlib import Path

from config.treehealth_5c import TreeHealthSegmentationPredictionConfig, TreeHealthSegmentationConfig
from predict.ordinal_watershed import OrdinalWatershedPredictor
from train.base import init_config
from train.treehealth_ordinal_watershed_5c import THOWatershedTrainer

if __name__ == '__main__':
    config_pred = init_config(TreeHealthSegmentationPredictionConfig())
    config_path = os.path.join(os.path.dirname(os.path.dirname(Path(config_pred.load))), 'config.json')
    with open(config_path) as f:
        config_train = json.loads(f.read())

    if config_pred.config_file is not None and config_pred.config_file != '' and os.path.exists(config_pred.config_file):
        with open(config_pred.config_file) as f:
            config_ = json.loads(f.read())
            config_pred.__dict__.update(config_)
            cls = config_pred

    config = init_config(cls)
    # Make sure the configuration for normalization is the same as the model loaded
    update_normalization = dict([(i, config_train.get(i)) for i in ['normalize', 'normalize_by_dataset', 'normalize_by_imagenet', 'in_channels', 'model_type', 'band_sequence']])
    # check if any of these values are none
    if any([i is None for i in update_normalization.values()]):
        raise ValueError("Normalization parameters are not set correctly.")
    config.__dict__.update(update_normalization)
    config.__dict__.update({'pretrained_ckpt': None})
    config.__dict__.update({'reset_head': False})

    config.__dict__.update({'save_samples': False})
    config.__dict__.update({'save_patch_df': False})
    config.__dict__.update({'save_idf': False})
    config.__dict__.update({'save_labeled_areas_df': False})
    config.__dict__.update({'processed_dir': None})

    config.__dict__.update({'allow_partial_patches': True})
    config.__dict__.update({'allow_empty_patches': True})

    config.__dict__.update({'n_edge_removal': 10})
    config.__dict__.update({'sequential_stride': config.patch_size - config.n_edge_removal * 2})

    config.__dict__.update({'split_col': None})
    config.__dict__.update({'dataset_split_by': 'patches'})

    config.__dict__.update({'image_srcs': {
        config.reference_source: {
            "base_path": Path("."),
            "image_file_type": ".tif",
            "image_file_prefix": "",
            "image_file_postfix": "",
            "filelist_path": "./deadtrees_images_to_predict.txt",
        },
    }})

    predictor = OrdinalWatershedPredictor(config, THOWatershedTrainer)
    os.makedirs(config.out_prediction_folder, exist_ok=True)
    predictor.predict(config.out_prediction_folder)
