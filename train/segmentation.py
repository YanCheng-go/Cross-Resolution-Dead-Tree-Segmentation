import logging
import os
import typing
import warnings

import geopandas as gpd
import torch
import torch.nn as nn
import torchmetrics

# from kornia.augmentation.random_generator import random_affine_generator
from kornia.geometry import hflip, vflip, get_affine_matrix2d, deg2rad, warp_affine
from tqdm import tqdm

from config.base import BaseSegmentationConfig
from src.data.base_dataset import train_val_test_split_data
from src.data.base_segmentation_dataset import BaseSegmentationDataset
from src.data.image_table import build_images_table
from src.modelling import helper
from src.modelling.helper import save_checkpoint
from src.utils.data_utils import stratified_random_split, is_same_crs
from src.utils.log_metrics import log_scalar, log_cmat_stats
from train.base import init_config, Trainer

warnings.filterwarnings("ignore")


class SegmentationTrainer(Trainer):
    Dataset = BaseSegmentationDataset

    def train_loop(self, epoch, epochs, n_train, train_loader, save_cp):
        self.aug_transform.train()
        self.model.train()
        train_loss = 0

        with tqdm(total=n_train * self.config.batch_size, desc=f'Epoch {epoch + 1}/{epochs}: train',
                  unit='img') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                x, y, m = self.aug_transform(batch, self.device)
                pred = self.model(x)
                if self.is_binary:  # Case of binary segmentation
                    pred_m = torch.masked_select(torch.squeeze(pred, dim=1), m)
                    pred_m = pred_m.reshape(1, -1)  # Bring back the batch dim (as 1)
                else:
                    sp = torch.swapaxes(pred, 0, 1)  # Swap channel and batch, needed for masked_select
                    pred_m = torch.masked_select(sp, m.unsqueeze(dim=0))  # for the swapped channel dim
                    pred_m = pred_m.reshape(1, pred.shape[1], -1)  # Bring back the batch (as 1) and channel dim

                y_m = torch.masked_select(y, m)
                y_m = y_m.reshape(1, -1)  # Bring back the batch dim

                loss = self.criterion(pred_m, y_m)
                loss.backward()
                train_loss += loss.item()

                # Accumulate gradients over multiple data loaders
                if (batch_idx + 1) % self.config.n_grad_accumulation == 0 or (batch_idx + 1 == n_train):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(x.shape[0])
                self.global_step += 1

                if epoch + 1 > self.config.lr_scheduler_warmup:
                    self.scheduler.step((epoch - self.config.lr_scheduler_warmup + batch_idx) / n_train)

                if batch_idx + 1 == n_train:
                    break

        train_loss /= n_train
        log_scalar(train_loss, f'loss/train', self.writer, self.global_step)
        log_scalar(self.optimizer.param_groups[0]['lr'], 'learning_rate', self.writer, self.global_step)

        # save the model
        save_checkpoint(save_cp, self.model, self.model_dir, self.optimizer, epoch, self.config.checkpoint_per_epoch)

        if self.verbose_logging and n_train > 0:
            self.images_to_tensorboard("train", x=x, y=y, pred=pred, m=m)

        return train_loss

    def evaluate(self, n, loader, name='val'):
        """Evaluation of the network"""
        self.aug_transform.eval()
        self.model.eval()
        val_loss = 0

        cc = 2 if self.is_binary else self.config.n_classes
        conf = torchmetrics.ConfusionMatrix(num_classes=cc).to(self.device)
        lpd = 0  # last_progress_digit for verbose visualization
        with tqdm(total=n * self.config.batch_size_val, desc=f'{name} round', unit='img') as pbar:
            for i, batch in enumerate(loader):
                x, y, m = self.aug_transform(batch, self.device)
                with torch.no_grad():
                    pred = self.model(x)
                    if self.is_binary:  # Case of binary segmentation
                        pred_m = torch.masked_select(torch.squeeze(pred, dim=1), m)
                        pred_m = pred_m.reshape(1, -1)  # Bring back the batch dim (as 1)
                        pred_arg = (pred_m > 0).squeeze()

                    else:
                        sp = torch.swapaxes(pred, 0, 1)  # Swap channel and batch, needed for masked_select
                        pred_m = torch.masked_select(sp, m.unsqueeze(dim=0))  # for the swapped channel dim
                        pred_m = pred_m.reshape(1, pred.shape[1], -1)  # Bring back the batch (as 1) and channel dim
                        pred_arg = pred_m.argmax(dim=1, keepdim=False)

                    y_m = torch.masked_select(y, m)
                    y_m = y_m.reshape(1, -1)  # Bring back the batch dim

                    loss = self.criterion(pred_m, y_m)
                    conf.update(pred_arg.flatten(), y_m.to(dtype=torch.int32).flatten())
                    progress = (
                            i * 10 // n)  # 0 - 9 if initialized properly
                    if self.verbose_logging and progress == lpd:
                        lpd += 1
                        self.images_to_tensorboard(f"{name}_{progress}", x=x, y=y, pred=pred, m=m)

                val_loss += loss.item()
                pbar.update(x.shape[0])
                if i + 1 == n:
                    break

        val_loss /= n
        stats = log_scalar(val_loss, f"loss/{name}", self.writer, self.global_step)

        cmat = conf.compute()
        stats.update(log_cmat_stats(cmat, name, self.writer, self.global_step, self.class_names))

        if self.verbose_logging and n > 0:
            self.images_to_tensorboard(name, x=x, y=y, pred=pred, m=m)

        out_metric = stats.get(self.config.get("val_metric", "loss/val"))

        return out_metric

    @classmethod
    def init_dataset_and_split(cls, config: BaseSegmentationConfig, labeled_areas, images_df, target_features,
                               **kwargs):
        assert target_features is not None

        if config.dataset_split_by == "patches":
            logging.info("Splitting patches into test, val and train")
            dataset = cls.Dataset.init_from_config(config, dataset_name="patches", labeled_areas=labeled_areas,
                                                   images_df=images_df, target_features=target_features, **kwargs)
            return train_val_test_split_data(
                config.val_split, config.test_split, dataset, None, None, rnds=11
            )
        elif config.dataset_split_by == "areas":
            logging.info("Splitting areas into test, val and train. In case of multiclass segmentation, " +
                         "we use stratified_random_split where " +
                         "areas are split based upon the count of various class instances.")
            train_areas, val_areas, test_areas = stratified_random_split(
                labeled_areas, images_df, config, target_features=target_features, write_split=False
            )

            train_dataset = cls.Dataset.init_from_config(config, "train", labeled_areas=train_areas,
                                                         images_df=images_df, target_features=target_features, **kwargs)
            val_dataset = cls.Dataset.init_from_config(config, "val", labeled_areas=val_areas, images_df=images_df,
                                                       target_features=target_features, **kwargs)
            test_dataset = cls.Dataset.init_from_config(config, "test", labeled_areas=test_areas, images_df=images_df,
                                                        target_features=target_features, **kwargs)

            return train_dataset, val_dataset, test_dataset
        else:
            raise Exception(
                f"No valid config for 'dataset_split_by': {config.dataset_split_by} (either 'patches' or 'areas'"
            )

    @staticmethod
    def init_data(config):

        images_df = build_images_table(
            config.image_srcs, config.reference_source, project_crs=config.project_crs,
            processed_dir=config.processed_dir, save_idf=config.save_idf
        )

        if config.labeled_areas_file is not None:
            areas = gpd.read_file(os.path.join(config.data_dir, config.labeled_areas_file))
        else:  # The whole is assumed to be labeled
            areas = images_df.query(f"src == @config.reference_source")[['geometry']].copy()

        if config.project_crs is None:
            config.project_crs = areas.crs
        elif not is_same_crs(areas.crs, config.project_crs):
            areas = areas.to_crs(config.project_crs)

        # Read in polygons
        target_features = gpd.read_file(os.path.join(config.data_dir, config.target_features_file))
        if not is_same_crs(target_features.crs, config.project_crs):
            target_features.to_crs(config.project_crs, inplace=True)

        data = {
            "labeled_areas": areas,
            "target_features": target_features,
            "images_df": images_df,
        }
        return data, config

    @staticmethod
    def init_model(config, device):
        return helper.initialize_model(config, device)

    @staticmethod
    def get_base_config():
        return BaseSegmentationConfig()

    @staticmethod
    def init_aug_transform(config):
        return SegAugmentation(config)


class SegAugmentation(nn.Module):
    def __init__(self, config, in_channels: typing.Union[int, typing.List[bool]] = None, norm_func=None) -> None:
        super(SegAugmentation, self).__init__()
        self.use_augmentation = config.use_augmentation
        self.base_config = config
        self.norm_func = norm_func
        self.max_translate = 0.00
        self.max_rotation = 179.
        self.max_shear = 0.00
        self.pixel_noise_chance = 0.25
        self.channel_noise_chance = 0.125
        self.pixel_drop_chance = 0.1
        self.pixel_drop_p = 0.01
        self.vflip_chance = 0.5
        self.hflip_chance = 0.5
        self.reference_source = config.reference_source
        if in_channels is None:
            in_channels = config.in_channels
        if type(in_channels) == int:
            self.channel_count = in_channels
            self.in_channels = [True] * in_channels
        else:
            self.channel_count = sum(in_channels)
            self.in_channels = in_channels
        self.channel_drop_chance = 1 / (self.channel_count * 2) if self.channel_count > 1 else 0.

        self.rotate_params_aug = nn.Parameter(torch.tensor(
            [-self.max_rotation / 2, self.max_rotation / 2]
        ), requires_grad=False)

        self.translate_params_aug = nn.Parameter(torch.tensor(
            [self.max_translate, self.max_translate]
        ), requires_grad=False)

        self.shear_params_aug = nn.Parameter(torch.tensor(
            [[-self.max_shear / 2, self.max_shear / 2],
             [-self.max_shear / 2, self.max_shear / 2]]
        ), requires_grad=False)

    def get_input(self, sample, device):
        if len(self.in_channels) < sample[self.reference_source].shape[1]:
            self.in_channels += [False] * (sample[self.reference_source].shape[1] - len(self.in_channels))
        x = sample[self.reference_source][:, self.in_channels].to(device)
        x = self.norm_func(x) if self.norm_func is not None else x
        x_m = ~torch.isnan(x)
        # fill nans with 0s
        x = torch.where(x_m, x, torch.tensor(0.0).to(device))
        return x

    @torch.no_grad()
    def forward(self, sample, device) -> object:
        x = self.get_input(sample, device)
        y = sample["target"].to(device).unsqueeze(1)
        m = sample["target_mask"].to(device).unsqueeze(1).bool()

        if self.use_augmentation and self.training:
            x, (y, m) = self.augment_(x, [y, m], x.shape, device)

        return x, y.squeeze(1), m.squeeze(1)

    def augment_(self, x, ys: list[torch.Tensor], reference_size, device):
        b, c, h, w = reference_size
        x = x.clone()
        ys = [y.clone() for y in ys]
        ysdype = [y.dtype for y in ys]
        mask = x != 0

        # pixelwise noise
        # additive noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .015
        x[mask] += ((torch.rand_like(x) * sigma) * noise_coin_n)[mask]
        # unoise
        sigma = .05
        x[mask] += ((torch.rand_like(x) * sigma) * noise_coin_u)[mask]

        # multiplicative noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005
        x[mask] += ((x * torch.rand_like(x) * sigma) * noise_coin_n)[mask]
        # unoise
        sigma = .015
        x[mask] += ((x * torch.rand_like(x) * sigma) * noise_coin_u)[mask]

        # channelwise noise
        # additive noise
        noise_coin = torch.rand((b, c, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_drop_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_drop_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .015
        x[mask] += ((torch.rand_like(x) * sigma) * noise_coin_n)[mask]
        # unoise
        sigma = .05
        x[mask] += ((torch.rand_like(x) * sigma) * noise_coin_u)[mask]

        # multiplicative noise
        noise_coin = torch.rand((b, c, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_drop_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_drop_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005
        x[mask] += ((x * torch.rand_like(x) * sigma) * noise_coin_n)[mask]
        # unoise
        sigma = .015
        x[mask] += ((x * torch.rand_like(x) * sigma) * noise_coin_u)[mask]

        # channel dropout
        x *= torch.floor(torch.rand((b, c, 1, 1), device=device) + (1 - self.channel_drop_chance))

        # pixel dropout
        x *= torch.clip(
            torch.floor(torch.rand((b, c, h, w), device=device) + (1 - self.pixel_drop_p)) +
            torch.floor(torch.rand((b, 1, 1, 1), device=device) + (1 - self.pixel_drop_chance)),
            max=1
        )

        # flipping horizontal
        hflip_coin = torch.floor(torch.rand((b, 1, 1, 1), device=device) + self.hflip_chance)
        x = x * (1 - hflip_coin) + hflip(x) * hflip_coin
        ys = [y * (1 - hflip_coin) + hflip(y) * hflip_coin for y in ys]

        # flipping vertical
        vflip_coin = torch.floor(torch.rand((b, 1, 1, 1), device=device) + self.vflip_chance)
        x = x * (1 - vflip_coin) + vflip(x) * vflip_coin
        ys = [y * (1 - vflip_coin) + vflip(y) * vflip_coin for y in ys]

        ys = [y.to(dtype=ydtype) for y, ydtype in zip(ys, ysdype)]
        return x, ys


if __name__ == '__main__':
    config = init_config(BaseSegmentationConfig())
    trainer = SegmentationTrainer(config)
    trainer.train_net()
