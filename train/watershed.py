import logging

import cv2
import numpy as np
import torch
import torchmetrics
from skimage.segmentation import watershed
from torch import nn
from torch.nn.functional import avg_pool2d, max_pool2d
from tqdm import tqdm

from config.base import BaseConfig
from config.watershed import WatershedConfig
from src.data.watershed_dataset import WatershedDataset
from src.modelling.helper import save_checkpoint
from src.modelling.losses.base import get_regression_loss, get_classification_loss
from src.modelling.models import UNet
from src.modelling.models.modules.convnext import CNBlock
from src.utils.jaccard import jaccard_from_confmat
from src.utils.log_metrics import log_scalar, log_cmat_stats
from src.visualization.visualize import plot_confusion_matrix
from train.segmentation import SegmentationTrainer, SegAugmentation

from kornia.geometry import hflip, vflip
from torch import nn, optim


class AttachConv(torch.nn.Module):
    def __init__(self, n_energy_bins: int, conv_layer: torch.nn.Conv2d, act_fn: torch.nn.Module,
                 norm_layer: torch.nn.Module):
        super().__init__()
        input_dim = conv_layer.weight.shape[1]
        cls_dim = conv_layer.weight.shape[0]
        self.seg_layer = conv_layer
        self.sobel_layer_pre = CNBlock(input_dim + cls_dim, 1, 0, act_fn, norm_layer)
        self.sobel_layer = torch.nn.Conv2d(input_dim + cls_dim, 2, 1, bias=False)
        self.ed_layer_pre = CNBlock(input_dim + cls_dim + 2, 1, 0, act_fn, norm_layer)
        self.energy_layer = torch.nn.Conv2d(input_dim + cls_dim + 2, n_energy_bins + 1, 1)
        self.density_layer = torch.nn.Conv2d(input_dim + cls_dim + 2, 1, 1, bias=False)

    def forward(self, x):
        seg = self.seg_layer(x)
        sobel_pre = self.sobel_layer_pre(torch.cat([x, seg.detach()], dim=1))
        sobel = self.sobel_layer(sobel_pre)
        sobel_x = torch.cat([sobel_pre, sobel.detach()], dim=1)
        energy = self.energy_layer(sobel_x)
        density = self.density_layer(sobel_x)

        return seg, sobel, energy, density


class WatershedTrainer(SegmentationTrainer):
    """closest reference: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf"""

    Dataset = WatershedDataset
    other_map = ["sobx", "soby", "energy", "density"]

    def __init__(self, config):
        super(WatershedTrainer, self).__init__(config)
        assert config.n_energy_bins > 2, "Watershed makes only sense for more than 2 n_energy_bins"
        self.criterion_sobel = get_regression_loss(config.loss_function_sobel.lower())
        self.criterion_density = get_regression_loss(config.loss_function_density.lower(), reduction="none")
        self.criterion_energy = get_classification_loss(config.loss_function_energy.lower())
        self.sobel_std = 2
        self.density_std = 0.1631
        self.count_std = 30

        self.to(self.device)

    @staticmethod
    def get_masked_tensor(x, m):
        sp = torch.swapaxes(x, 0, 1)  # Swap channel and batch, needed for masked_select
        x_m = torch.masked_select(sp, m.unsqueeze(dim=0))  # for the swapped channel dim
        # Bring back the batch (as 1) and channel dim
        return x_m.reshape(1, x.shape[1], -1)

    def train_loop(self, epoch, epochs, n_train, train_loader, save_cp):
        self.aug_transform.train()
        self.model.train()
        train_loss = 0
        train_loss_seg = 0
        train_loss_sobel = 0
        train_loss_energy = 0
        train_loss_density = 0
        train_loss_count = 0

        with tqdm(total=n_train * self.config.batch_size, desc=f'Epoch {epoch + 1}/{epochs}: train',
                  unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                x, y_seg, y_sobel, y_energy, y_density, m = self.aug_transform(batch, self.device)
                pred_seg, pred_sobel, pred_energy, pred_density = self.model(x)

                if self.is_binary:  # Case of binary segmentation
                    pred_seg_m = torch.masked_select(pred_seg.squeeze(1), m)
                    # Bring back the batch dim (as 1)
                    pred_seg_m = pred_seg_m.reshape(1, -1)
                else:
                    pred_seg_m = pred_seg_m(pred_seg, m)

                y_seg_m = torch.masked_select(y_seg, m)
                y_seg_m = y_seg_m.reshape(1, -1)  # Bring back the batch dim

                seg_loss = self.criterion(pred_seg_m, y_seg_m)

                # integral loss over different sizes
                m_f = m.float()
                kernel_size = (3, 5, 7)

                y_count = y_density.squeeze(1).sum((1, 2))
                pred_count = (pred_density.squeeze(1) * m).sum((1, 2))
                count_loss = abs(pred_count - y_count).mean()
                # # over entire image
                density_loss = (self.criterion_density(pred_count, y_count) / (m.sum((1, 2)) + 1e-6)).mean()

                # # over each pixel
                pred_density_m = torch.masked_select(pred_density.squeeze(1), m)
                # Bring back the batch dim (as 1)
                pred_density_m = pred_density_m.reshape(1, -1)

                y_density_m = torch.masked_select(y_density.squeeze(1), m)
                y_density_m = y_density_m.reshape(1, -1)  # Bring back the batch dim

                density_loss += self.criterion_density(pred_density_m, y_density_m).mean()

                for ks in kernel_size:
                    y_density_ks = avg_pool2d(y_density, ks, divisor_override=1)
                    pred_density_ks = avg_pool2d(pred_density, ks, divisor_override=1)
                    mm_ks = max_pool2d(m_f, ks).bool()
                    mka_ks = avg_pool2d(m_f, ks, divisor_override=1)

                    pred_density_m = torch.masked_select(pred_density_ks.squeeze(1) / (mka_ks + 1e-6), mm_ks)
                    # Bring back the batch dim (as 1)
                    pred_density_m = pred_density_m.reshape(1, -1)

                    y_density_m = torch.masked_select(y_density_ks.squeeze(1) / (mka_ks + 1e-6), mm_ks)
                    y_density_m = y_density_m.reshape(1, -1)  # Bring back the batch dim

                    ## normalized by number of pixels in ks
                    density_loss += (self.criterion_density(pred_density_m, y_density_m)).mean()

                # sobel loss
                pred_sobel_m = self.get_masked_tensor(pred_sobel, m)
                y_sobel_m = self.get_masked_tensor(y_sobel, m)

                sobel_loss = self.criterion_sobel(pred_sobel_m, y_sobel_m)

                # energy loss
                pred_energy_m = self.get_masked_tensor(pred_energy, m)
                y_energy = torch.clamp(y_energy, max=self.config.n_energy_bins)

                y_energy_m = self.get_masked_tensor(y_energy, m).squeeze(1)
                energy_loss = self.criterion_energy(pred_energy_m, y_energy_m)

                loss = seg_loss \
                       + self.config.loss_density_weight * (density_loss / self.count_std) \
                       + self.config.loss_sobel_weight * (sobel_loss / self.sobel_std) \
                       + self.config.loss_energy_weight * energy_loss
                train_loss += loss.item()
                train_loss_seg += seg_loss.item()
                train_loss_count += count_loss.item()
                train_loss_sobel += sobel_loss.item()
                train_loss_density += density_loss.item()
                train_loss_energy += energy_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix(**{'loss': loss.item()})
                pbar.update(x.shape[0])
                self.global_step += 1

                if epoch + 1 > self.config.lr_scheduler_warmup:
                    if self.config.lr_scheduler in ["steps", "multi_steps"]:
                        self.scheduler.step(epoch)
                    else:
                        self.scheduler.step((epoch - self.config.lr_scheduler_warmup + i) / n_train)

                if i + 1 == n_train:
                    break

        train_loss /= n_train
        train_loss_seg /= n_train
        train_loss_sobel /= n_train
        train_loss_density /= n_train
        train_loss_energy /= n_train
        train_loss_count /= n_train

        log_scalar(train_loss, f'loss/train', self.writer, self.global_step)
        log_scalar(train_loss_seg, f'loss_seg/train', self.writer, self.global_step)
        log_scalar(train_loss_sobel, f'loss_sobel/train', self.writer, self.global_step)
        log_scalar(train_loss_density, f'loss_density/train', self.writer, self.global_step)
        log_scalar(train_loss_energy, f'loss_energy/train', self.writer, self.global_step)
        log_scalar(train_loss_count, f'loss_count/train', self.writer, self.global_step)
        log_scalar(self.optimizer.param_groups[0]['lr'], 'learning_rate', self.writer, self.global_step)

        # save the model
        save_checkpoint(save_cp, self.model, self.model_dir, self.optimizer, epoch, self.config.checkpoint_per_epoch)

        if self.verbose_logging:
            self.images_to_tensorboard(
                "train", pred=pred_seg, m=m, x=x, y=y_seg, y_sobel=y_sobel, pred_sobel=pred_sobel,
                y_density=y_density, pred_density=pred_density, y_energy=y_energy, pred_energy=pred_energy
            )

        return train_loss

    def evaluate(self, n, loader, name='val'):
        """Evaluation of the network"""
        self.aug_transform.eval()
        self.model.eval()
        val_loss = 0
        val_loss_seg = 0
        val_loss_sobel = 0
        val_loss_energy = 0
        val_loss_density = 0
        val_loss_density_count = 0
        val_loss_wcount = 0

        cc = 2 if self.is_binary else self.config.n_classes
        conf = torchmetrics.ConfusionMatrix(task='binary', num_classes=cc).to(self.device)
        conf_watershed = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device)
        lpd = 0  # last_progress_digit for verbose visualization
        with tqdm(total=n * self.config.batch_size_val, desc=f'{name} round', unit='img') as pbar:
            for i, batch in enumerate(loader):
                with torch.no_grad():
                    x, y_seg, y_sobel, y_energy, y_density, m = self.aug_transform(batch, self.device)
                    pred_seg, pred_sobel, pred_energy, pred_density = self.model(x)

                    # get object count
                    y_count = batch.get("count", None)
                    if y_count is None:
                        # DEPRECATED this is only necessary for old dataset versions
                        count_mask = ((y_energy != 0) * m.unsqueeze(1)).cpu().numpy()
                        y_count = np.array(
                            [len(np.unique(
                                watershed(-y_energy[i].cpu().numpy(), connectivity=2, mask=count_mask[i])
                            )) - 1 for i in range(m.shape[0])]
                        ).astype(float)
                    else:
                        y_count = y_count.numpy()

                    if self.is_binary:  # Case of binary segmentation
                        pred_seg_m = torch.masked_select(pred_seg.squeeze(1), m)
                        # Bring back the batch dim (as 1)
                        pred_seg_m = pred_seg_m.reshape(1, -1)
                        pred_arg = (pred_seg_m > 0).squeeze()
                    else:
                        pred_seg_m = pred_seg_m(pred_seg, m)
                        pred_arg = pred_seg_m.argmax(dim=1, keepdim=False)

                    y_seg_m = torch.masked_select(y_seg, m)
                    y_seg_m = y_seg_m.reshape(1, -1)  # Bring back the batch dim

                    seg_loss = self.criterion(pred_seg_m, y_seg_m)

                    y_count_d = y_density.squeeze(1).sum((1, 2))
                    pred_count = (pred_density.squeeze(1) * m).sum((1, 2))
                    count_loss = abs(pred_count - y_count_d).mean()

                    density_loss = (self.criterion_density(pred_count, y_count_d) / (m.sum((1, 2)) + 1e-6)).mean()

                    energy = pred_energy.argmax(1)
                    count_mask = ((energy > 0) * m).cpu().numpy()
                    watershed_count = np.array(
                        [len(np.unique(
                            watershed(-energy[i].cpu().numpy(), connectivity=2, mask=count_mask[i])
                        )) - 1 for i in range(m.shape[0])]
                    ).astype(float)
                    wcount_loss = abs(watershed_count - y_count).mean()

                    # # over each pixel
                    pred_density_m = torch.masked_select(pred_density.squeeze(1), m)
                    # Bring back the batch dim (as 1)
                    pred_density_m = pred_density_m.reshape(1, -1)

                    y_density_m = torch.masked_select(y_density.squeeze(1), m)
                    y_density_m = y_density_m.reshape(1, -1)  # Bring back the batch dim

                    density_loss += self.criterion_density(pred_density_m, y_density_m).mean()
                    kernel_size = (3, 5, 7)

                    m_f = m.float()
                    for ks in kernel_size:
                        y_density_ks = avg_pool2d(y_density, ks, divisor_override=1)
                        pred_density_ks = avg_pool2d(pred_density, ks, divisor_override=1)
                        mm_ks = max_pool2d(m_f, ks).bool()
                        mka_ks = avg_pool2d(m_f, ks, divisor_override=1)

                        pred_density_m = torch.masked_select(pred_density_ks.squeeze(1) / (mka_ks + 1e-6), mm_ks)
                        # Bring back the batch dim (as 1)
                        pred_density_m = pred_density_m.reshape(1, -1)

                        y_density_m = torch.masked_select(y_density_ks.squeeze(1) / (mka_ks + 1e-6), mm_ks)
                        y_density_m = y_density_m.reshape(1, -1)  # Bring back the batch dim

                        ## normalized by number of pixels in ks
                        density_loss += (self.criterion_density(pred_density_m, y_density_m)).mean()

                    # sobel loss
                    pred_sobel_m = self.get_masked_tensor(pred_sobel, m)
                    y_sobel_m = self.get_masked_tensor(y_sobel, m)

                    sobel_loss = self.criterion_sobel(pred_sobel_m, y_sobel_m)

                    # energy loss
                    pred_energy_m = self.get_masked_tensor(pred_energy, m)
                    # # get energy levels up to n_energy_bins, if there are more, add them to class 1
                    y_energy = torch.clamp(y_energy, max=self.config.n_energy_bins)

                    y_energy_m = self.get_masked_tensor(y_energy, m).squeeze(1)
                    energy_loss = self.criterion_energy(pred_energy_m, y_energy_m)

                    loss = seg_loss \
                           + self.config.loss_density_weight * (density_loss / self.count_std) \
                           + self.config.loss_sobel_weight * (sobel_loss / self.sobel_std) \
                           + self.config.loss_energy_weight * energy_loss

                    val_loss += loss.item()
                    val_loss_seg += seg_loss.item()
                    val_loss_density_count += count_loss.item()
                    val_loss_wcount += wcount_loss
                    val_loss_sobel += sobel_loss.item()
                    val_loss_density += density_loss.item()
                    val_loss_energy += energy_loss.item()
                    conf.update(pred_arg.flatten(), y_seg_m.to(dtype=torch.int32).flatten())

                    conf_watershed.update(
                        (pred_energy_m.argmax(dim=1) > 0).flatten(), y_seg_m.to(dtype=torch.int32).flatten()
                    )
                    # 0 - 9 if initialized properly
                    progress = (i * 10 // n)
                    if self.verbose_logging and progress == lpd:
                        lpd += 1
                        self.images_to_tensorboard(
                            name, pred=pred_seg, m=m, x=x, y=y_seg, y_sobel=y_sobel, pred_sobel=pred_sobel,
                            y_density=y_density, pred_density=pred_density, y_energy=y_energy, pred_energy=pred_energy
                        )

                pbar.update(x.shape[0])
                if i + 1 == n:
                    break

        val_loss /= n
        val_loss_seg /= n
        val_loss_sobel /= n
        val_loss_density /= n
        val_loss_energy /= n
        val_loss_density_count /= n
        val_loss_wcount /= n

        # logging losses
        stats = log_scalar(val_loss, f"loss/{name}", self.writer, self.global_step)
        stats.update(log_scalar(val_loss_sobel, f"loss_sobel/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_energy, f"loss_energy/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_density, f"loss_density/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_density_count, f"loss_density_count/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount, f"loss_count/{name}", self.writer, self.global_step))

        cmat = conf.compute()
        stats.update(log_cmat_stats(cmat, name, self.writer, self.global_step, self.class_names))
        cmat_watershed = conf_watershed.compute()
        stats.update(
            log_cmat_stats(cmat_watershed, f"watershed_/{name}", self.writer, self.global_step, self.class_names)
        )

        if self.verbose_logging:
            self.images_to_tensorboard(
                name, pred=pred_seg, m=m, x=x, y=y_seg, y_sobel=y_sobel, pred_sobel=pred_sobel,
                y_density=y_density, pred_density=pred_density, y_energy=y_energy, pred_energy=pred_energy
            )

        out_metric = stats.get(self.config.get("val_metric", "loss/val"))

        return out_metric

    def log_segmentation_performance(self, cc, cmat, name):
        miou = jaccard_from_confmat(cmat, num_classes=cc).item()
        tp = cmat[torch.eye(cc, dtype=torch.bool)].sum().item()
        numel = cmat.sum().item()
        acc = tp / numel
        recall = (cmat[torch.eye(cc, dtype=torch.bool)] /
                  (cmat.sum(0, keepdim=True) + torch.finfo(torch.float32).eps)).mean().item()
        f1 = 2 * ((acc * recall) / (acc + recall)) if acc + recall != 0 else 0.0
        logging.info(f"accuracy/{name}: {acc}")
        logging.info(f"tp/{name}: {tp}")
        logging.info(f"recall/{name}: {recall}")
        logging.info(f"mIOU/{name}: {miou}")
        logging.info(f"f1/{name}: {f1}")
        logging.info(f"confusion_matrix/{name}: {cmat}")
        self.writer.add_scalar(f"accuracy/{name}", acc, self.global_step)
        self.writer.add_scalar(f"tp/{name}", tp, self.global_step)
        self.writer.add_scalar(f"recall/{name}", recall, self.global_step)
        self.writer.add_scalar(f"mIOU/{name}", miou, self.global_step)
        self.writer.add_scalar(f"f1/{name}", f1, self.global_step)
        # Done this way, since not sure how normalize = True effect mious etc
        cmatn = cmat / cmat.sum(axis=1, keepdim=True)
        cmatn = np.nan_to_num(cmatn.cpu().numpy())
        self.writer.add_figure(f"{name}/confusion_matrix", plot_confusion_matrix(cmatn, self.class_names))

    @classmethod
    def init_model(cls, config, device):
        model, optimizer = WatershedUNet.load_from_config(config, device)

        return model, optimizer

    @staticmethod
    def get_bn_classase_config():
        return WatershedConfig()

    @staticmethod
    def init_aug_transform(config):
        return WatershedAugmentation(config)

    def images_to_tensorboard(self, name, x=None, y=None, pred=None, m=None, pred_sobel=None, pred_density=None,
                              pred_energy=None, y_sobel=None, y_density=None, y_energy=None):
        super().images_to_tensorboard(name, x, y, pred, m)
        if y_sobel is not None:
            nmin = -1
            nmax = 1
            y_sobel = (y_sobel - nmin) / (nmax - nmin)
            self.writer.add_images(f'{name}/target_sobx', y_sobel[:, [0]], self.global_step)
            self.writer.add_images(f'{name}/target_soby', y_sobel[:, [1]], self.global_step)
        if pred_sobel is not None:
            nmin = -1
            nmax = 1
            pred_sobel = (pred_sobel - nmin) / (nmax - nmin)
            self.writer.add_images(f'{name}/prediction_sobx', pred_sobel[:, [0]], self.global_step)
            self.writer.add_images(f'{name}/prediction_soby', pred_sobel[:, [1]], self.global_step)
        if y_density is not None:
            nmax = y_density.amax()
            y_density = y_density / nmax
            self.writer.add_images(f'{name}/target_density', y_density, self.global_step)
            if pred_density is not None:
                pred_density = pred_density / nmax
                self.writer.add_images(f'{name}/prediction_density', pred_density, self.global_step)
        if y_energy is not None:
            y_energy = y_energy.float() / self.config.n_energy_bins
            self.writer.add_images(f'{name}/target_energy', y_energy, self.global_step)
        if pred_energy is not None:
            pred_energy = pred_energy.argmax(dim=1, keepdim=True).float() / self.config.n_energy_bins
            self.writer.add_images(f'{name}/prediction_energy', pred_energy, self.global_step)


class WatershedUNet(UNet):
    def __init__(self, n_energy_bins, **kwargs):
        super().__init__(**kwargs)
        self.n_energy_bins = n_energy_bins
        self.last_conv = AttachConv(
            self.n_energy_bins, self.last_conv, self.activation_layer, self.norm_layer_down
        )

    @classmethod
    def get_model_dict_from_config(cls, st_dict):
        model_dict = super().get_model_dict_from_config(st_dict)
        model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
        return model_dict

    @classmethod
    def get_save_dict(cls, model, optm=None):
        st_dict = super().get_save_dict(model, optm)
        st_dict["n_energy_bins"] = model.n_energy_bins
        return st_dict


class WatershedAugmentation(SegAugmentation):

    @torch.no_grad()
    def forward(self, sample, device, dataset_name='patch') -> object:
        x = self.get_input(sample, device, dataset_name=dataset_name)
        y = sample["target"].to(device).unsqueeze(1)
        sobel = torch.stack([sample["sobx"], sample["soby"]], dim=1).to(device)
        energy = sample["energy"].unsqueeze(1).long().to(device)
        density = sample["density"].unsqueeze(1).to(device)
        m = sample["target_mask"].to(device).unsqueeze(1).bool()

        ori_res = sample["ori_resolution"]
        ha = sample["hectares"]
        patch_res = ori_res.clone()
        patch_ids = sample["patch_id"]

        count_weights = sample['count_weights'].unsqueeze(1).to(device) if 'count_weights' in sample.keys() else torch.ones_like(m).to(device)
        edge_weights = sample['edge_weights'].unsqueeze(1).to(device) if 'edge_weights' in sample.keys() else torch.ones_like(m).to(device)

        self.remove_edge_on_the_fly = False  # use this when data in the processed_dir does not have inner_mask band
        self.reverse_padding = 10
        self.remove_edge = True  # when there is inner mask band in the processed_dir
        if 'inner_mask' in sample.keys() and self.remove_edge and not self.remove_edge_on_the_fly:
            m = sample["inner_mask"].to(device).unsqueeze(1).bool()
            sample["count"] = sample["inner_count"]

        ycount = sample['count']

        # prediction will not use augmentation
        # train and evaluation and test will use augmentation, but only evaluation and test with resampling need ycount
        if self.use_augmentation:
            if self.base_config.rescale:
                scale_factors = sample["scale_factors"].numpy()
            else:
                scale_factors = None
            if not self.training:
                x, [y, sobel, energy, density, m, count_weights, edge_weights, ycount, ha, patch_res, idxs, scales] = self.augment_(x, [y, sobel, energy, density, m, count_weights, edge_weights, ycount, patch_ids, ha, ori_res], x.shape, device, scale_factors, self.training, manual_resample=False, dataset_name=dataset_name)
                return x, y.squeeze(1), sobel, energy, density, m.squeeze(1), count_weights, edge_weights, ycount, ha, patch_res, idxs, scales
            elif self.training:
                x, [y, sobel, energy, density, m, count_weights, edge_weights, ha, patch_res, scales] = self.augment_(x, [y, sobel, energy, density, m, count_weights, edge_weights, patch_ids, ha, ori_res], x.shape, device, scale_factors, self.training, manual_resample=False, dataset_name=dataset_name)
                return x, y.squeeze(1), sobel, energy, density, m.squeeze(1), count_weights, edge_weights, ha, patch_res, scales
        else:
            if not self.training:
                return x, y.squeeze(1), sobel, energy, density, m.squeeze(
                    1), count_weights, edge_weights, ha, patch_res, torch.arange(0, x.shape[0], device=device), torch.ones_like(ycount).to(device)
            else:
                return x, y.squeeze(1), sobel, energy, density, m.squeeze(1), count_weights, edge_weights, ha, patch_res, torch.ones_like(ycount).to(device)


    def augment_(self, x, ys: list[torch.Tensor], reference_size, device):
        b, c, h, w = reference_size
        x = x.clone()
        ys = [y.clone() for y in ys]
        ysdype = [y.dtype for y in ys]
        mask = x != 0

        # pixelwise noise (each image has a chance of having random noise per pixel)
        # additive noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .015
        x += ((torch.randn_like(x).clip(-3, 3) * sigma) * noise_coin_n) * mask
        # unoise
        sigma = .05
        x += ((torch.rand_like(x) * sigma) * noise_coin_u) * mask

        # multiplicative noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.pixel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.pixel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005
        x += ((x * torch.randn_like(x).clip(-3, 3) * sigma) * noise_coin_n) * mask
        # unoise
        sigma = .015
        x += ((x * torch.rand_like(x) * sigma) * noise_coin_u) * mask

        # channelwise noise (each image has a chance of having random noise per channel)
        # additive noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .015
        x += ((torch.randn((b, c, 1, 1), device=device).clip(-3, 3) * sigma) * noise_coin_n) * mask
        # unoise
        sigma = .05
        x += ((torch.rand((b, c, 1, 1), device=device) * sigma) * noise_coin_u) * mask

        # multiplicative noise
        noise_coin = torch.rand((b, 1, 1, 1), device=device)
        noise_coin_u = (noise_coin < self.channel_noise_chance / 2)
        noise_coin_n = ((noise_coin < self.channel_noise_chance) & ~noise_coin_u).float()
        noise_coin_u = noise_coin_u.float()
        # nnoise
        sigma = .005
        x += ((x * torch.randn((b, c, 1, 1), device=device).clip(-3, 3) * sigma) * noise_coin_n) * mask
        # unoise
        sigma = .015
        x += ((x * torch.rand((b, c, 1, 1), device=device) * sigma) * noise_coin_u) * mask

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
