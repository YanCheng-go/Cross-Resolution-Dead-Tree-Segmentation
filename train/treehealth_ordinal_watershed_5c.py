import json
import os
from pathlib import Path

import kornia

# if kornia.__version__ != '0.7.0':
#     raise NotImplementedError("Update kornia to version 0.7.0")

from config.treehealth_5c import TreeHealthSegmentationConfig
from train.base import init_config
from train.treehealth_ordinal_watershed import THOWatershedTrainer, THOWatershedAugmentation


class THOWatershedTrainer_5c(THOWatershedTrainer):
    @staticmethod
    def init_aug_transform(config):
        return THOWatershedAugmentation_5c(config)


class THOWatershedAugmentation_5c(THOWatershedAugmentation):
    def __init__(self, config):
        THOWatershedAugmentation.__init__(self, config)


if __name__ == '__main__':
    config_pred = init_config(TreeHealthSegmentationConfig())

    if config_pred.config_file is not None and config_pred.config_file != '' and os.path.exists(config_pred.config_file):
        with open(config_pred.config_file) as f:
            config_ = json.loads(f.read())
            config_pred.__dict__.update(config_)
            cls = config_pred

    else:
        if config_pred.load is not None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(Path(config_pred.load))), 'config.json')
            with open(config_path) as f:
                config_train = json.loads(f.read())
            cls = init_config(TreeHealthSegmentationConfig())
            updated = dict([(i, config_train.get(i)) for i in cls.__dict__ if cls.__dict__.get(i)
                            != config_train.get(i) and i != '__doc__'])
            cls = TreeHealthSegmentationConfig()
            cls.__dict__.update(updated)

            # Add configs from the cml
            config = init_config(cls)
            # Make sure the configuration for normalization is the same as the model loaded
            update_normalization = dict([(i, config_train.get(i)) for i in
                                         ['normalize', 'normalize_by_dataset', 'normalize_by_imagenet', 'in_channels',
                                          'model_type', 'band_sequence']])
            # check if any of these values are none
            if any([i is None for i in update_normalization.values()]):
                raise ValueError("Normalization parameters are not set correctly.")
            config.__dict__.update(update_normalization)
            config.__dict__.update({'pretrained_ckpt': None})
            config.__dict__.update({'reset_head': False})

            # check if reset_head is True
            if config.reset_head:
                raise ValueError("You are continue training a model or test this model with reset_head as True. Please double check the configuration.")

            cls = config

        else:
            cls = TreeHealthSegmentationConfig()

    config = init_config(cls)
    trainer = THOWatershedTrainer_5c(config)
    trainer.train_net()
