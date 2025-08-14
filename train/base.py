import json
import logging
import os
import random
from functools import partial
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset
from tensorboardX import SummaryWriter

from config.base import BaseConfig
from src.data.base_dataset import get_dataloader, BaseDataset
from src.data.collate import default_collate_with_shapely_support
from src.modelling import helper
from src.modelling.helper import clean_epoch_checkpoints, model_params_as_str
from src.modelling.losses.base import get_binary_classification_loss, get_classification_loss
from src.utils import config_parser
from src.visualization.visualize import image_for_display


def set_torch_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_config(config: BaseConfig, testing_mode: bool = False) -> BaseConfig:
    set_torch_seeds(config.seed)
    # load config and parse args
    if not testing_mode:
        parser = config_parser.config_to_argparser(config)
        args = parser.parse_args()
        config_parser.update_conf_with_parsed_args(config, args)

    if config.pipeline_test:
        config.batch_size = 2
        config.batch_size_val = 2

    config_check(config)
    return config


normalize_for_tensorboard = partial(image_for_display, per_instance=True)


def get_soft_pred(config, pred):
    if config.n_classes == 1:
        soft_pred = pred.sigmoid()
    else:
        soft_pred = pred.softmax(1)
    return soft_pred


def config_check(config):
    assert config.project_crs is not None, "project_crs must be set"
    if hasattr(config, 'dataset_split_by'):
        if config.dataset_split_by == "patches":
            assert config.val_split > 0 and config.test_split > 0


class Trainer(nn.Module):
    Dataset: BaseDataset = None

    def __init__(self, config):
        assert self.Dataset is not None

        super(Trainer, self).__init__()

        params = model_params_as_str(config)
        if config.run_dir_suffix is None:
            self.run_dir = helper.initialize_rundir(config.log_dir, params)
        else:
            self.run_dir = helper.initialize_rundir(config.log_dir, config.run_dir_suffix)
        helper.save_config_to_rundir(self.run_dir, config)

        self.model_dir = os.path.join(self.run_dir, 'model')
        self.best_model_path = os.path.join(self.model_dir, 'BestModel.pth')
        self.tensorboard_dir = os.path.join(self.run_dir, 'tensorboard')
        self.writer = None
        self.global_step = 0
        self.verbose_logging = config.verbose_logging
        # initialize data and add project crs if not provided
        data, config = self.init_data(config)
        assert config.target_classes is None or \
               len(config.target_classes) == config.n_classes or \
               (len(config.target_classes) == 2 and config.n_classes == 1), \
            "Different class count found. \n" \
            "Possible reason:" \
            "\t Config class names ('target_classes') and " \
            "config number of classes ('n_classes') do not align "

        train_dataset, val_dataset, test_dataset = self.init_dataset_and_split(config, **data)

        # # add information on the automatic split in patch_grids.shp
        # patch_df = BaseDataset.patch_df
        # if self.split_col is None:
        #     if "auto_split" not in patch_df.columns:
        #         patch_df['auto_split'] = ''
        #         patch_df.iloc[train_dataset.indices.tolist(), -1] = 'train'
        #         patch_df.iloc[val_dataset.indices.tolist(), -1] = 'val'
        #         patch_df.iloc[test_dataset.indices.tolist(), -1] = 'test'
        #         p = self.processed_dir / 'qgis' / self.dataset_name
        #         p.mkdir(exist_ok=True, parents=True)
        #         patch_df[['geometry', 'area_id', 'auto_split']].to_file(p / "patch_grid_autoSplit.gpkg", driver="GPKG")

        self.train_loader = get_dataloader(
            train_dataset, config.batch_size, config.num_workers,
            collate_fn=default_collate_with_shapely_support, train=True
        )
        self.val_loader = get_dataloader(
            val_dataset, config.batch_size_val, config.num_workers,
            collate_fn=default_collate_with_shapely_support, train=False
        )

        self.test_loader = get_dataloader(
            test_dataset, config.batch_size_val, config.num_workers,
            collate_fn=default_collate_with_shapely_support, train=False
        )

        if isinstance(self.train_loader.dataset, Subset):
            self.class_names = getattr(
                self.train_loader.dataset.dataset,
                "target_classes", list(range(config.n_classes))
            )
        else:
            if hasattr(self.train_loader.dataset, "target_classes"):
                self.class_names = self.train_loader.dataset.target_classes
            else:
                self.class_names = None
        assert self.class_names is None or config.n_classes == len(self.class_names) or \
               (config.n_classes == 1 and len(self.class_names) == 2), \
            "Number of configured classes and dataset classes are not the same. \n" \
            "Possible reason: \n" \
            "\t Derived class names from labels is different from config class number."
        if hasattr(config, "device"):
            self.device = helper.get_device() if config.device is None else config.device
        else:
            self.device = helper.get_device()
        logging.info(f'Using device {self.device}')

        # initialize augmentation and preprocessing
        self.aug_transform = self.init_aug_transform(config)

        # initialize model
        self.model, self.optimizer = self.init_model(config, self.device)
        logging.info(self.model)

        # initialize training related things
        self.is_binary = config.n_classes == 1
        self.criterion = self.get_loss(config.loss_function.lower())

        if config.lr_scheduler == "cosinewr":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 10, 2)
        elif config.lr_scheduler == 'multi_steps':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [50, 300, 1000], 0.1) # change the values
        elif config.lr_scheduler == 'steps':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 50, 0.1)
        elif config.lr_scheduler == "none" or config.lr_scheduler is None:
            class NoScheduler():
                def step(self, epoch): return

            self.scheduler = NoScheduler()
        else:
            raise Exception(f"lr scheduler {config.lr_scheduler} not implemented.")

        assert config.val_metric_opt in ["min", "max"]
        self.metric_opt = np.less if config.val_metric_opt == "min" else np.greater

        self.config = config

        self.to(self.device)

    def get_loss(self, loss):
        loss = loss.lower()
        if self.is_binary:
            return get_binary_classification_loss(loss)
        else:
            return get_classification_loss(loss)

    def train_net(self, save_cp=True):
        """start training process for number of epochs according to config given at initialization"""
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self.writer.add_text('config', json.dumps(self.config._dump(), indent=4, sort_keys=True), 0)
        if self.config.pipeline_test:
            n_train = min(1, len(self.train_loader))
            n_val = min(1, len(self.val_loader))
            n_test = min(1, len(self.test_loader))
            epochs = 1
        else:
            n_train = len(self.train_loader)
            n_val = len(self.val_loader)
            n_test = len(self.test_loader)
            epochs = self.config.epochs

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {self.config.batch_size}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            run_dir:         {self.run_dir}
        ''')
        if self.config.warmup > 0:
            self.model.freeze(backbone=True, other=True, head=False)
        elif self.config.backbone_warmup > 0:
            self.model.freeze(backbone=True, other=False, head=False)

        lowest_loss = None
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch + 1} ...")

            if (self.config.backbone_warmup == epoch != 0) or (self.config.warmup == epoch != 0):
                if self.config.backbone_warmup > self.config.warmup != 0 and self.config.warmup == epoch:
                    logging.info("everything except the backbone is now being trained")
                    self.model.freeze(backbone=True, other=False, head=False)
                else:
                    logging.info("entire network is now being trained")
                    self.model.freeze(backbone=False, other=False, head=False)

            self.train_loop(epoch, epochs, n_train, self.train_loader, save_cp)

            # Validation at the end of epoch
            val_loss = self.evaluate(n_val, self.val_loader, "val")

            # check if we found a better model than before
            if lowest_loss is None or self.metric_opt(val_loss, lowest_loss):
                lowest_loss = val_loss

                clean_epoch_checkpoints(self.best_model_path, epoch, self.model_dir)

                # save new best model
                self.model.save_model(self.model, self.best_model_path, optm=self.optimizer)

                self.writer.add_text(
                    "Training/New Best Model",
                    f"New best model saved at epoch {epoch + 1}  "
                    f"\n New Best Validation Loss: {val_loss}  ",
                    self.global_step
                )
                logging.info("     | > Saved as new best model.")
                logging.info('Saved the best model, found in epoch: {}'.format(epoch + 1))
                logging.info('New Best Validation Loss: {}'.format(val_loss))

        # make a final evaluation on the test set
        if epochs == 0:
            self.best_model_path = self.config.load
            self.use_augmentation = False

        self.model.load_from_path(self.best_model_path, self.device)
        self.evaluate(n_test, self.test_loader, "test")

        self.writer.close()

    def train_loop(self, epoch, epochs, n_train, train_loader, save_cp):
        """Here the model is learning from the train_loader for one epoch"""
        raise NotImplementedError("function train_loop is not implemented")

    def evaluate(self, n, loader, name='val'):
        """Evaluating the model on a given loader. Returning val loss."""
        raise NotImplementedError("function evaluate is not implemented")

    def images_to_tensorboard(self, name, x=None, y=None, pred=None, m=None):
        if x is not None:
            self.writer.add_images(f'{name}/images', normalize_for_tensorboard(x[:, :3]), self.global_step)
            # for i in range(x.shape[1]):
            #     self.writer.add_histogram(f'{name}/images_band_{i}', torch.flatten(x[:, i]), self.global_step)
        if m is not None:
            if len(m.shape) == 3:
                m = m.unsqueeze(1)
            self.writer.add_images(f'{name}/mask', m, self.global_step)
        if self.is_binary:
            if y is not None:
                for i, cls in enumerate(self.class_names):
                    self.writer.add_images(
                        f'{name}/target_{cls}', y.float().unsqueeze(1), self.global_step,
                        dataformats="NCHW"
                    )
            if pred is not None:
                soft_pred = get_soft_pred(self.config, pred)
                for i, cls in enumerate(self.class_names):
                    self.writer.add_images(
                        f'{name}/prediction_{cls}', soft_pred, self.global_step,
                        dataformats="NCHW"
                    )
        else:
            if y is not None:
                for i, cls in enumerate(self.class_names):
                    self.writer.add_images(
                        f'{name}/target_{cls}', (y == i).float().unsqueeze(1), self.global_step,
                        dataformats="NCHW"
                    )
            if pred is not None:
                soft_pred = get_soft_pred(self.config, pred)
                for i, cls in enumerate(self.class_names):
                    self.writer.add_images(
                        f'{name}/prediction_{cls}', soft_pred[:, [i]], self.global_step,
                        dataformats="NCHW"
                    )

    @staticmethod
    def init_data(config: BaseConfig) -> Tuple[dict, BaseConfig]:
        """loading idf, areas, and maybe polygons as dictionary"""
        raise NotImplementedError("function init_data is not implemented")

    @classmethod
    def init_dataset_and_split(cls, config: BaseConfig, areas, idf, polygons=None) -> Tuple[
        BaseDataset, BaseDataset, BaseDataset]:
        """return train/val/test datasets for given data"""
        raise NotImplementedError("function init_dataset_and_split is not implemented")

    @staticmethod
    def init_model(config: BaseConfig, device) -> Tuple[nn.Module, optim.Optimizer]:
        """loading model and optimizer"""
        return helper.initialize_model(config, device)

    @staticmethod
    def get_base_config() -> BaseConfig:
        """get the base config for this task"""
        raise NotImplementedError("function get_base_config is not implemented")
