import itertools
import os
from copy import copy

import ipdb
import numpy as np
import pandas as pd
import torch
import torchmetrics
from skimage.segmentation import watershed
from torch.nn.functional import one_hot
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from src.data.base_dataset import get_dataloader
from src.data.collate import default_collate_with_shapely_support
from src.modelling import helper
from src.modelling.helper import save_checkpoint, model_params_as_str
from src.modelling.losses.base import get_binary_classification_loss
from src.modelling.models import UNet
from src.modelling.models import PretrainedUNet
# from src.modelling.models.MANet_with_scalar import MANetWithScalar
from src.modelling.models.unet_with_scalar import UNetWithScalar
# from src.modelling.models.unet_with_scalar_v2 import UNetWithScalarV2
# from src.modelling.models.unet_with_biome import UNetWithGeofuse
from src.utils.log_metrics import log_scalar, log_cmat_stats
from train.base import normalize_for_tensorboard
from train.watershed import WatershedTrainer

import json
import logging
from tensorboardX import SummaryWriter
from src.modelling.helper import clean_epoch_checkpoints
import seaborn as sns

import matplotlib.pyplot as plt
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['nature', 'science'])


def plot_residuals(lis_watershed_count, lis_y_count, ha_factor=1):
    x, y = np.array(lis_watershed_count), (np.array(lis_y_count) - np.array(lis_watershed_count))

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams.update({'font.size': 14})
    g = sns.jointplot(x=x * ha_factor, y=y * ha_factor, legend=False, ax=ax)
    g.refline(y=0)
    # sns.move_legend(g.ax_joint, "lower right", bbox_to_anchor=(1.1, 0.8), title='', frameon=True)
    g.set_axis_labels('Predicted count per ha', 'Residuals')

    g.fig.set_figwidth(5.6)
    g.fig.set_figheight(5.6)
    plt.tight_layout()
    plt.show()


def energy_stats(y_energy_all, set_name='all', n_bins=30, sep_viz=False, fig_fp=None):
    """Export statistics for energy levels"""

    if set_name == "all":
        y_energy = torch.cat(list(y_energy_all.values()))
    else:
        y_energy = y_energy_all[set_name]

    y_energy_gt0 = y_energy[y_energy != 0.0].float().cpu()
    mean_, std_, max_ = y_energy_gt0.mean(), y_energy_gt0.std(), y_energy_gt0.max()
    print(f"The mean, std, max values of energy levels for {set_name} set are: {mean_}, {std_}, {max_}")

    # Plot histograms for the energy levels
    if set_name == "all" and sep_viz is True:
        for idx, i in enumerate(list(y_energy_all.values())):
            hist = torch.histc(i[i != 0.0].float().cpu(), bins=n_bins, min=1, max=n_bins)
            plt.bar([i + idx * 0.2 for i in list(range(n_bins))], hist, align="center", width=0.2,
                    label=list(y_energy_all.keys())[idx])
    else:
        hist = torch.histc(y_energy_gt0, bins=n_bins, min=1, max=n_bins)
        plt.bar(range(n_bins), hist, align="center", label=set_name)

    plt.xticks(range(n_bins), [i + 1 for i in range(n_bins)])
    plt.xlabel("Energy level")
    plt.ylabel("Number of pixels")
    plt.title("")
    plt.legend()
    plt.show()


def viz_batch(x_batch, y_batch, batch_id=None, cmap_scale=(0, 20), fig_fp=None, n_cols=4, n_rows=4):
    """Visualize batches"""

    batch_size = len(x_batch)
    n_rows = 2 * 2 if n_rows is None else n_rows
    n_cols = int(batch_size/n_rows * 2 + 0.5) if batch_size == 1 else int(batch_size / n_rows * 2) if n_cols is None else n_cols
    f, axarr = plt.subplots(n_rows, int(n_cols), squeeze=False)

    images = []

    for idx, (n_row, n_col) in enumerate(product(range(n_rows), range(n_cols))):
        if idx < batch_size:
            axarr[n_row][n_col].imshow(x_batch[idx].cpu().permute(1, 2, 0).numpy()[:, :, :-1])
        else:
            images.append(axarr[n_row][n_col].imshow(y_batch[idx - batch_size].squeeze().cpu().numpy(), cmap="Oranges",
                          vmin=cmap_scale[0], vmax=cmap_scale[1]))
    f.suptitle(f"The raw images and energy layers for batch_id={batch_id}")
    f.colorbar(images[0], ax=axarr, orientation='horizontal', fraction=.1)
    if fig_fp is not None:
        plt.savefig(fig_fp, dpi=300)
    else:
        plt.show()


def generate_mp(mi=3, mx=11, interval=2, up_lim=99):

    """
    mp output example:
    # mp = {
    #     3: [3, 5],
    #     4: [5, 7],
    #     5: [7, 9],
    #     6: [9, 11],
    #     7: [11, 13],
    #     8: [13, 15],
    #     9: [15, 17],
    #     10: [17, 19],
    #     11: [19, 99]
    # }
    """

    assert mi > 0

    mx = (mx - mi) * interval + mi
    a = list(np.arange(mi, mx + 1, interval))
    b = a[1:]
    b.extend([up_lim])
    vals = list(zip(a, b))
    mp = dict(zip(list(np.arange(mi, mx + 1, 1)), vals))

    return mp


def mp_func(x, mp_dict):
    if x < 0:
        return None
    if x < min(list(mp_dict.keys())):
        return x
    for k, v in mp_dict.items():
        if v[0] <= x <= v[1]:
            return k
    return -1


class AttachConv(torch.nn.Module):
    def __init__(self, n_energy_bins: int, conv_layer: torch.nn.Conv2d, ordinal_connect: bool):
        super().__init__()
        input_dim = conv_layer.weight.shape[1]
        cls_dim = conv_layer.weight.shape[0]
        assert cls_dim == 1
        self.sobel_layer = torch.nn.Conv2d(input_dim, 2, 1, bias=False)
        self.energy_layer = torch.nn.Conv2d(input_dim, n_energy_bins, 1)
        self.ordinal_connect = ordinal_connect

    def forward(self, x):
        energy = self.energy_layer(x)
        if self.ordinal_connect:
            energy = energy.cumsum(1)

        return self.sobel_layer(x), energy


def weightedSampler(dataset, split_by="patches"):
    """According to Igel's method"""

    if split_by == "patches":
        train_patches = copy(dataset.dataset.patch_df.iloc[dataset.indices])
    else:
        train_patches = copy(dataset.patch_df)

    assert "weight_vars" in train_patches.columns
    assert train_patches['weight_vars'].isnull().sum() == 0

    # check the unique value of weight_vars and remove empty or minority..
    # e.g. assert train_patches['spatial_clusters'].isnull().sum() == 0. Otherwise, replace NaN in the dataframe
    weight_vars = np.unique(train_patches['weight_vars'])
    weight_vars = weight_vars[0] if len(weight_vars) == 1 else None
    weight_col = []

    def calc_weights(x, n_classes, n_patches, c=0.9):
        """x: number of patches per class"""
        return c * (1 / n_classes) * (1 / x) + (1 - c) * (1 / n_patches)

    train_patches = pd.DataFrame(train_patches)
    n_patches = len(train_patches)
    if "spatial_clusters" in weight_vars:
        train_patches['clusterCount'] = train_patches.groupby('spatial_clusters')['area_id'].transform('size')
        train_patches['cluster_weight'] = calc_weights(
            train_patches['clusterCount'], len(train_patches['spatial_clusters'].unique()), n_patches
        )
        weight_col.append('cluster_weight')
    if "patch_hectares" in weight_vars or 'ori_resolution' in weight_vars:
        train_patches["ori_resolution"] = train_patches.apply(lambda x: round(x["ori_transform"][0], 3), axis=1)
        train_patches['resCount'] = train_patches.groupby('ori_resolution')['area_id'].transform('size')
        train_patches['resolution_weight'] = calc_weights(
            train_patches['resCount'], len(train_patches['ori_resolution'].unique()), n_patches
        )
        weight_col.append('resolution_weight')
    if 'patch_hectares' in weight_vars and 'feature_counts' in weight_vars:
        # Count per hectare...
        train_patches['density'] = train_patches['feature_counts'] / train_patches['patch_hectares']
        train_patches['density_class'] = 0
        train_patches['density_weight'] = 0
        pd.np.histogram(np.sqrt(train_patches[train_patches['density'] != 0]['density']), bins=30)
        train_patches.loc[train_patches['density'] != 0, ['density_class']] = (
            pd.cut(np.sqrt(train_patches[train_patches['density'] != 0]['density']), bins=30, labels=range(1, 31)))
        train_patches['densityCount'] = train_patches.groupby('density_class')['area_id'].transform('size')
        train_patches.loc[train_patches['density'] == 0, 'density_weight'] = (
                0.01 / len(train_patches[train_patches['density'] == 0]))
        train_patches.loc[train_patches['density'] != 0, 'density_weight'] = 0.99 * calc_weights(
            train_patches[train_patches['density'] != 0]['densityCount'],
            len(train_patches[train_patches['density'] != 0]['density_class'].unique()),
            len(train_patches[train_patches['density'] != 0])
        )
        weight_col.append('density_weight')
    train_patches['weight_aggregated'] = train_patches[weight_col].sum(axis=1)
    train_patches['weight_aggregated'] = train_patches['weight_aggregated'] / train_patches['weight_aggregated'].sum()

    sampler = WeightedRandomSampler(train_patches['weight_aggregated'].tolist(), len(dataset), replacement=True)
    return sampler


class OrdinalWatershedTrainer(WatershedTrainer):
    """closest reference: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf"""

    def __init__(self, config):
        super(OrdinalWatershedTrainer, self).__init__(config)
        self.criterion_energy = get_binary_classification_loss(config.loss_function_energy.lower())
        self.mp_dict = generate_mp(config.fuse_mi, config.fuse_mx, config.fuse_interval)
        self.f = lambda x: mp_func(x, self.mp_dict)

        # # statistics for energy levels
        # y_energy_ = y_energy
        # self.y_energy_train = []
        # self.y_energy_val = []
        # self.y_energy_test = []
        # self.y_energy_all = []

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

        # Use the weighted sampler if the config is set to True
        sampler = weightedSampler(train_dataset, self.config.dataset_split_by) if config.weighted_sampling else None

        self.train_loader = get_dataloader(
            train_dataset, config.batch_size, config.num_workers,
            collate_fn=default_collate_with_shapely_support, train=True,
            sampler=sampler, shuffle=False
        )

    other_map = ["sobx", "soby", "energy"]
    val_metrics = ['loss', 'loss_wcount', 'loss_wcount_cumprod', 'loss_sobel', 'loss_energy']

    def train_loop(self, epoch, epochs, n_train, train_loader, save_cp):
        self.aug_transform.train()
        self.model.train()
        train_loss = 0
        train_loss_sobel = 0
        train_loss_energy = 0

        with tqdm(total=n_train * self.config.batch_size, desc=f'Epoch {epoch + 1}/{epochs}: train',
                  unit='img') as pbar:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, batch in enumerate(train_loader):
                # # check pre- and post-augmentation per batch
                # # import ipdb;ipdb.set_trace()
                # self.aug_transform.eval()
                # x, y_seg, y_sobel, y_energy, y_density, m = self.aug_transform(batch, self.device)
                # viz_batch(x_batch=x, y_batch=y_energy, batch_id=i)
                # self.aug_transform.train()

                if self.config.split_col is None:
                    dataset_name = 'patches'
                else:
                    dataset_name = 'train'

                # batch = batch[batch['hectares'] < 1 / 4 * pow(batch['ori_resolution'] * 256, 2)]

                x, y_seg, y_sobel, y_energy, y_density, m, count_weights, edge_weights, _, patch_res, scales = self.aug_transform(batch, self.device, dataset_name=dataset_name)

                self.config.remove_edge_on_the_fly = False
                if self.config.remove_edge_on_the_fly is True:
                    non_zeros = torch.nonzero(m, as_tuple=True)
                    row_start, col_start = non_zeros[1][self.get_indices_from_mask(non_zeros)], non_zeros[2][
                        self.get_indices_from_mask(non_zeros)]
                    non_zeros2 = torch.unbind(torch.flip(torch.stack(non_zeros, dim=1), [0]), dim=1)
                    row_end, col_end = non_zeros2[1][self.get_indices_from_mask(non_zeros2)], non_zeros2[2][
                        self.get_indices_from_mask(non_zeros2)]
                    reverse_padding = torch.tensor(self.config.reverse_padding * scales, device=self.device,
                                                   dtype=torch.uint8)  # 10 is a magic number
                    m_stride = torch.zeros_like(m)
                    for idd in range(m.shape[0]):
                        m_stride[idd, row_start[idd] + reverse_padding[idd]:row_end[idd] - reverse_padding[idd],
                        col_start[idd] + reverse_padding[idd]:col_end[idd] - reverse_padding[idd]] = 1
                    m = m_stride.clone()

                if self.config.fuse_in is True:
                    y_energy = torch.from_numpy(np.vectorize(self.f)(y_energy.cpu())).to(self.device)

                additional_scalars = [patch_res.to(self.device)]
                # viz_batch(x_batch=x, y_batch=y_energy, batch_id=i)
                if 'with_scalar' in self.config.model_type:
                    pred_sobel, pred_energy = self.model(x, additional_scalars)
                # elif 'geofuse' in self.config.model_type:
                #     geo_scalar = one_hot()
                #     pred_sobel, pred_energy = self.model(x, additional_scalars, geo_scalar)
                else:
                    pred_sobel, pred_energy = self.model(x)

                # if self.config.fuse_out is True:
                #     pred_energy = torch.from_numpy(np.vectorize(self.f)(pred_energy.cpu())).to(self.device)

                # # statistics for energy levels
                # y_energy_ = y_energy

                # sobel loss
                pred_sobel_m = self.get_masked_tensor(pred_sobel, m)
                y_sobel_m = self.get_masked_tensor(y_sobel, m)

                sobel_loss = self.criterion_sobel(pred_sobel_m, y_sobel_m)

                # energy loss
                pred_energy_m = self.get_masked_tensor(pred_energy, m)
                # import ipdb;ipdb.set_trace()
                y_energy = torch.clamp(y_energy, max=self.config.n_energy_bins)
                y_energy = one_hot(y_energy, self.config.n_energy_bins + 1)[..., 1:]
                y_energy = y_energy.flip(-1)
                y_energy.cumsum_(-1)
                y_energy = y_energy.flip(-1)
                y_energy.swapaxes_(1, -1)
                y_energy.squeeze_(-1)

                y_energy_m = self.get_masked_tensor(y_energy, m).squeeze(1)

                weights = torch.ones_like(m).unsqueeze(1).to(self.device)
                if self.config.apply_count_weights:
                    weights = (count_weights * weights).to(self.device)
                    # count_weights_m = self.get_masked_tensor(count_weights, m)
                if self.config.apply_edge_weights:
                    weights = (edge_weights * weights).to(self.device)
                    # edge_weights_m = self.get_masked_tensor(edge_weights_m, m)

                weights_m = self.get_masked_tensor(weights, m)
                energy_loss = self.criterion_energy(pred_energy_m, y_energy_m, weights_m)

                loss = self.config.loss_sobel_weight * (sobel_loss / self.sobel_std) \
                       + self.config.loss_energy_weight * energy_loss

                train_loss += loss.item()
                train_loss_sobel += sobel_loss.item()
                train_loss_energy += energy_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.model.scaler.scale(loss).backward()
                # self.model.scaler.step(self.optimizer)
                # self.model.scaler.update()

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

                # # statistics for energy levels
                # self.y_energy_train.append(y_energy_)

                # res_info = int(
                #     "".join([str(int(res * 10)) for res in
                #              patch_res.numpy()]))  # 10 is a constant depending on the resolution range...
                # log_scalar(res_info, f"patch_resolutions/train", self.writer, self.global_step)

        train_loss /= n_train
        train_loss_sobel /= n_train
        train_loss_energy /= n_train
        log_scalar(train_loss, f'loss/train', self.writer, self.global_step)
        log_scalar(train_loss_sobel, f'loss_sobel/train', self.writer, self.global_step)
        log_scalar(train_loss_energy, f'loss_energy/train', self.writer, self.global_step)
        log_scalar(self.optimizer.param_groups[0]['lr'], 'learning_rate', self.writer, self.global_step)

        # save the model
        save_checkpoint(save_cp, self.model, self.model_dir, self.optimizer, epoch, self.config.checkpoint_per_epoch)

        if self.verbose_logging:
            pred_energy.masked_fill_(~m.unsqueeze(1), 0)  # ensuring that outside areas are empty
            self.images_to_tensorboard(
                "train", m=m, x=normalize_for_tensorboard(x), y=y_seg, y_sobel=y_sobel, pred_sobel=pred_sobel, y_energy=y_energy,
                pred_energy=pred_energy
            )

        torch.cuda.empty_cache()
        return train_loss

    def get_indices_from_mask(self, non_zeros):
        unique, idx, counts = torch.unique(non_zeros[0], sorted=False, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=self.device), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        return first_indicies

    def evaluate(self, n, loader, name='val'):
        """Evaluation of the network"""
        self.aug_transform.eval()
        self.model.eval()
        val_loss = 0
        val_loss_sobel = 0
        val_loss_energy = 0
        val_loss_wcount = 0
        val_loss_wcount_cumprod = 0
        val_loss_wcount_comb = 0
        val_loss_cumprod_comb = 0
        total_n = 0
        val_loss_wcount_cumprod_0 = 0
        val_loss_wcount_cumprod_1 = 0
        val_loss_wcount_cumprod_10 = 0

        # lis_val_loss_wcount = []
        # lis_val_loss_wcount_cumprod = []
        # lis_watershed_count = []
        # lis_y_count = []
        # lis_watershed_count_cumprob = []

        conf_watershed = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device)
        conf_watershed_cumprod = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device)
        lpd = 0  # last_progress_digit for verbose visualization
        with tqdm(total=n * self.config.batch_size_val, desc=f'{name} round', unit='img') as pbar:
            for i, batch in enumerate(loader):
                with torch.no_grad():
                    # batch = batch[batch['hectares'] < 1 / 4 * pow(batch['ori_resolution'] * 256, 2)]

                    x, y_seg, y_sobel, y_energy, _, m, count_weights, edge_weights, ycount, ha, patch_res, idxs, scales = self.aug_transform(batch, self.device, dataset_name=name)

                    self.config.remove_edge_on_the_fly = False
                    # Remove edge effects in the evaluation, when the segmentation polygons are on the edge of the patch
                    if self.config.remove_edge_on_the_fly is True:
                        non_zeros = torch.nonzero(m, as_tuple=True)
                        row_start, col_start = non_zeros[1][self.get_indices_from_mask(non_zeros)], non_zeros[2][self.get_indices_from_mask(non_zeros)]

                        non_zeros2 = torch.unbind(torch.flip(torch.stack(non_zeros, dim=1), [0]), dim=1)
                        row_end, col_end = non_zeros2[1][self.get_indices_from_mask(non_zeros2)], non_zeros2[2][self.get_indices_from_mask(non_zeros2)]

                        reverse_padding = torch.tensor(self.config.reverse_padding * scales, device=self.device, dtype=torch.uint8) # 10 is a magic number

                        m_stride = torch.zeros_like(m)
                        for idd in range(m.shape[0]):
                            m_stride[idd, row_start[idd]+reverse_padding[idd]:row_end[idd]-reverse_padding[idd], col_start[idd]+reverse_padding[idd]:col_end[idd]-reverse_padding[idd]] = 1
                            ycount[idd] = (torch.unique(y_seg[idd] * (m[idd] * m_stride[idd]), return_counts=True)[0] != 0).sum()

                        m = m_stride.clone()

                    if self.config.fuse_in is True:
                        y_energy = torch.from_numpy(np.vectorize(self.f)(y_energy.cpu())).to(self.device)

                    y_seg[y_seg != 0] = 1
                    # y_energy_ = y_energy
                    additional_scalars = [patch_res.to(self.device)]
                    # viz_batch(x_batch=x, y_batch=y_energy, batch_id=i)
                    if 'with_scalar' in self.config.model_type:
                        pred_sobel, pred_energy = self.model(x, additional_scalars)
                    else:
                        pred_sobel, pred_energy = self.model(x)

                    # get object count
                    # y_count = batch.get("count", None)
                    y_count = ycount.clone()
                    del ycount
                    if y_count is None:
                        # DEPRECATED this is only necessary for old dataset versions
                        count_mask = ((y_energy != 0) * m.unsqueeze(1)).cpu().numpy()
                        y_count = np.array(
                            [len(np.unique(
                                watershed(-y_energy[i].cpu().numpy(), connectivity=2, mask=count_mask[i])
                            )) - 1 for i in range(m.shape[0])]
                        ).astype(float)
                        # y_count = torch.tensor(
                        #     [len([j for j in torch.unique(y_seg_m[i]) if j != 0]) for i in range(m.shape[0])]).to(
                        #     self.device)

                    else:
                        y_count = y_count.numpy()

                    # sobel loss
                    pred_sobel_m = self.get_masked_tensor(pred_sobel, m)
                    y_sobel_m = self.get_masked_tensor(y_sobel, m)

                    sobel_loss = self.criterion_sobel(pred_sobel_m, y_sobel_m)

                    # energy loss
                    pred_energy_m = self.get_masked_tensor(pred_energy, m)
                    # # get energy levels up to n_energy_bins, if there are more, add them to class 1
                    y_energy = torch.clamp(y_energy, max=self.config.n_energy_bins)
                    y_energy = one_hot(y_energy, self.config.n_energy_bins + 1)[..., 1:]
                    y_energy = y_energy.flip(-1)
                    y_energy.cumsum_(-1)
                    y_energy = y_energy.flip(-1)
                    y_energy.swapaxes_(1, -1)
                    y_energy.squeeze_(-1)

                    y_energy_m = self.get_masked_tensor(y_energy, m).squeeze(1)

                    weights = torch.ones_like(m).unsqueeze(1).to(self.device)
                    if self.config.apply_count_weights:
                        weights = (count_weights * weights).to(self.device)
                    if self.config.apply_edge_weights:
                        weights = (edge_weights * weights).to(self.device)

                    weights_m = self.get_masked_tensor(weights, m)
                    energy_loss = self.criterion_energy(pred_energy_m, y_energy_m, weights_m)

                    loss = self.config.loss_sobel_weight * (sobel_loss / self.sobel_std) \
                           + self.config.loss_energy_weight * energy_loss

                    y_seg_m = torch.masked_select(y_seg, m)
                    y_seg_m = y_seg_m.reshape(1, -1)  # Bring back the batch dim

                    pred_energy.masked_fill_(~m.unsqueeze(1), 0)  # ensuring that outside areas are empty
                    pred_energy_ = (pred_energy > 0).float().sum(1)
                    pred_energy_cumprod = (pred_energy > 0).float().cumprod(1).sum(1)

                    if self.config.fuse_out is True:
                        pred_energy_ = torch.from_numpy(np.vectorize(self.f)(pred_energy_.cpu())).to(self.device)
                        pred_energy_cumprod = torch.from_numpy(np.vectorize(self.f)(pred_energy_cumprod.cpu())).to(self.device)

                    count_mask = ((pred_energy_ != 0) * m).cpu().numpy()
                    watershed_count = np.array(
                        [len(np.unique(
                            watershed(-pred_energy_[i].cpu().numpy(), connectivity=2, mask=count_mask[i])
                        )) - 1 for i in range(m.shape[0])]
                    ).astype(float)

                    per_ha = False  # Test whether to use per ha...
                    if not per_ha:
                        ha_copy = torch.ones_like(ha)
                    else:
                        ha_copy = ha.clone()
                    wcount_loss = abs(np.array([a / b for a, b in zip(watershed_count - y_count, ha_copy) if b != 0])).mean()
                    wcount_loss_1 = abs(np.array([a / b for a, b in zip(watershed_count[y_count != 0] - y_count[y_count != 0], ha_copy[y_count != 0]) if b != 0])).mean() if (y_count != 0).sum() != 0 else 0
                    wcount_loss_0 = abs(np.array([a / b for a, b in zip(watershed_count[y_count == 0] - y_count[y_count == 0], ha_copy[y_count == 0]) if b != 0])).mean() if (y_count == 0).sum() != 0 else 0

                    # if np.isnan(wcount_loss):
                    #     import ipdb;
                    #     ipdb.set_trace()
                    # else:
                    #     continue

                    count_mask = ((pred_energy_cumprod != 0) * m).cpu().numpy()
                    watershed_count_cumprob = np.array(
                        [len(np.unique(
                            watershed(-pred_energy_cumprod[i].cpu().numpy(), connectivity=2, mask=count_mask[i])
                        )) - 1 for i in range(m.shape[0])]
                    ).astype(float)

                    wcount_loss_cumprod = abs(np.array([a / b for a, b in zip(watershed_count_cumprob - y_count, ha_copy) if b != 0])).mean()
                    wcount_loss_cumprod_1 = abs(np.array([a / b for a, b in zip(watershed_count_cumprob[y_count != 0] - y_count[y_count != 0], ha_copy[y_count != 0]) if b != 0])).mean() if (y_count != 0).sum() != 0 else 0
                    wcount_loss_cumprod_0 = abs(np.array([a / b for a, b in zip(watershed_count_cumprob[y_count == 0] - y_count[y_count == 0], ha_copy[y_count == 0]) if b != 0])).mean() if (y_count == 0).sum() != 0 else 0

                    scales = scales.cpu().numpy()
                    if self.config.use_augmentation and self.config.auto_resample:
                        wcount_loss_cumprod_10 = abs(np.array([a / b for a, b in zip(
                            watershed_count_cumprob[(y_count != 0) & (scales == 1)] - y_count[(y_count != 0) & (scales == 1)], ha_copy[(y_count != 0) & (scales == 1)]) if
                                                              b != 0])).mean() if (y_count != 0).sum() != 0 else 0 # no rescale...
                    else:
                        wcount_loss_cumprod_10 = 0

                    pred_energy_m = self.get_masked_tensor(pred_energy_.unsqueeze(1), m)
                    conf_watershed.update(
                        (pred_energy_m != 0).flatten(),
                        y_seg_m.to(dtype=torch.int32).flatten()
                    )

                    pred_energy_m = self.get_masked_tensor(pred_energy_cumprod.unsqueeze(1), m)
                    conf_watershed_cumprod.update(
                        (pred_energy_m != 0).flatten(),
                        y_seg_m.to(dtype=torch.int32).flatten()
                    )

                    total_n += x.shape[0]

                    val_loss += loss.item()
                    val_loss_wcount += wcount_loss
                    val_loss_wcount_cumprod += wcount_loss_cumprod
                    val_loss_wcount_cumprod_0 += wcount_loss_cumprod_0
                    val_loss_wcount_cumprod_1 += wcount_loss_cumprod_1
                    val_loss_wcount_cumprod_10 += wcount_loss_cumprod_10
                    val_loss_sobel += sobel_loss.item()
                    val_loss_energy += energy_loss.item()
                    loss_cumprod_comb = energy_loss.item() * 1000 + wcount_loss_cumprod
                    val_loss_cumprod_comb += loss_cumprod_comb
                    loss_wcount_comb = energy_loss.item() * 1000 + wcount_loss
                    val_loss_wcount_comb += loss_wcount_comb

                    if self.config.epochs == 0:
                        # lis_val_loss_wcount.append(wcount_loss)
                        # lis_val_loss_wcount_cumprod.append(wcount_loss_cumprod)
                        # lis_watershed_count.extend(watershed_count.tolist())
                        # lis_watershed_count_cumprob.extend(watershed_count_cumprob.tolist())
                        # lis_y_count.extend(y_count.tolist())
                        lis_watershed_count = 1
                        lis_y_count = 1

                    # 0 - 9 if initialized properly
                    progress = (i * 10 // n)
                    if self.verbose_logging and progress == lpd:
                        lpd += 1
                        self.images_to_tensorboard(
                            name, m=m, x=normalize_for_tensorboard(x), y=y_seg, y_sobel=y_sobel, pred_sobel=pred_sobel,
                            y_energy=y_energy, pred_energy=pred_energy
                        )

                    # # Note down the resolution information?
                    # res_info = ",".join([str(int(res * 100)) for res in patch_res])
                    # with open("patch_resolutions_cm.txt", 'w') as txtfile:
                    #     txtfile.write(f"global_step: {self.global_step} {res_info}\n")

                # # statistics for energy levels
                # if name == 'val':
                #     self.y_energy_val.append(y_energy_)
                # elif name == 'test':
                #     self.y_energy_test.append(y_energy_)

                pbar.update(x.shape[0])
                if i + 1 == n:
                    break

                n = copy(total_n)

        val_loss /= n
        val_loss_sobel /= n
        val_loss_energy /= n
        val_loss_wcount /= n
        val_loss_wcount_cumprod /= n
        val_loss_wcount_cumprod_0 /= n
        val_loss_wcount_cumprod_1 /= n
        val_loss_wcount_cumprod_10 /= n

        # logging losses
        stats = log_scalar(val_loss, f"loss/{name}", self.writer, self.global_step)
        stats.update(log_scalar(val_loss_sobel, f"loss_sobel/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_energy, f"loss_energy/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount, f"loss_count/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount_cumprod, f"loss_count_cumprod/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount_cumprod_0, f"loss_count_cumprod_0/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount_cumprod_1, f"loss_count_cumprod_1/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount_cumprod_10, f"loss_count_cumprod_10/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_cumprod_comb, f"loss_cumprod_comb/{name}", self.writer, self.global_step))
        stats.update(log_scalar(val_loss_wcount_comb, f"loss_wcount_comb/{name}", self.writer, self.global_step))

        cmat_watershed = conf_watershed.compute()
        stats.update(log_cmat_stats(cmat_watershed, f"watershed/{name}",
                                    self.writer, self.global_step, self.class_names))
        cmat_watershed_cumprod = conf_watershed_cumprod.compute()
        stats.update(log_cmat_stats(cmat_watershed_cumprod, f"watershed_cumprod/{name}",
                                    self.writer, self.global_step, self.class_names))

        # Plot residuals of count per ha... only for the testing round
        plot_res = False
        if self.config.epochs == 0 and plot_res:
            res = 0.2
            ha_factor = (100 / (self.config.patch_size * res)) ** 2
            plot_residuals(lis_watershed_count, lis_y_count, ha_factor=ha_factor)

        if self.verbose_logging:
            self.images_to_tensorboard(
                name, m=m, x=normalize_for_tensorboard(x), y=y_seg, y_sobel=y_sobel, pred_sobel=pred_sobel,
                y_energy=y_energy, pred_energy=pred_energy
            )

        out_metric = stats.get(self.config.get("val_metric", "loss/val"))

        torch.cuda.empty_cache()
        return out_metric

    def images_to_tensorboard(self, name, x=None, y=None, pred=None, m=None, pred_sobel=None, pred_density=None,
                              pred_energy=None, y_sobel=None, y_density=None, y_energy=None):
        super().images_to_tensorboard(
            name, x=x, y=y, pred=None, m=m, pred_sobel=pred_sobel, y_energy=None, pred_energy=None
        )
        if y_energy is not None:
            y_energy = y_energy.sum(1, keepdim=True).float()
            # y_energy /= self.config.n_energy_bins
            self.writer.add_images(f'{name}/target_energy', normalize_for_tensorboard(y_energy), self.global_step)
        if pred_energy is not None:
            pred_energy_ = (pred_energy > 0).float().sum(1, keepdim=True)
            # pred_energy_ /= self.config.n_energy_bins
            self.writer.add_images(f'{name}/prediction_energy', normalize_for_tensorboard(pred_energy_), self.global_step)

            pred_energy_cumprod = (pred_energy > 0).float().cumprod(1).sum(1, keepdim=True)
            # pred_energy_cumprod /= self.config.n_energy_bins
            self.writer.add_images(f'{name}/prediction_energy_cumprod', normalize_for_tensorboard(pred_energy_cumprod), self.global_step)

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

        if epochs == -1:
            self.best_model_path = self.config.load
            val_loss = self.evaluate(n_val, self.val_loader, "val")
            self.writer.close()

        # make a final evaluation on the test set
        if epochs == 0:
            self.best_model_path = self.config.load
        self.model.load_from_path(self.best_model_path, self.device)
        self.evaluate(n_test, self.test_loader, "test")

        # # statistics for energy levels
        # self.y_energy_all = {'train': torch.cat(self.y_energy_train),
        #                      'val': torch.cat(self.y_energy_val),
        #                      'test': torch.cat(self.y_energy_test)}
        # import ipdb;ipdb.set_trace()
        # energy_stats(y_energy_all=self.y_energy_all, set_name="all", sep_viz=True)

        self.writer.close()

    @classmethod
    def init_model(cls, config, device):
        if config.model_type == "pretrained_unet":
            model, optimizer = PretrainedWatershedUNet.load_from_config(config, device)
        # elif config.model_type == "pretrained_with_scalar":
        #     model, optimizer = WatershedMANetWithScalar.load_from_config(config, device)
        elif config.model_type == "unet_with_scalar":
            model, optimizer = WatershedUNetWithScalar.load_from_config(config, device)
        # elif config.model_type == "unet_with_scalar_v2":
        #     model, optimizer = WatershedUNetWithScalarV2.load_from_config(config, device)
        # elif config.model_type == "unet_with_geofuse":
        #     model, optimizer = WatershedUNetWithGeofuse.load_from_config(config, device)
        else:
            model, optimizer = WatershedUNet.load_from_config(config, device)
        return model, optimizer


class WatershedUNet(UNet):
    def __init__(self, n_energy_bins: int, ordinal_connect: bool, **kwargs):
        # add scaler_weights,
        super().__init__(**kwargs)

        # self.scaler = torch.cuda.amp.GradScaler()
        # if scaler_weights is not None:
        #     self.scaler.load_state_dict(scaler_weights)
        self.n_energy_bins = n_energy_bins
        self.last_conv = AttachConv(self.n_energy_bins, self.last_conv, ordinal_connect=ordinal_connect)

    @classmethod
    def get_model_dict_from_config(cls, st_dict):
        model_dict = super().get_model_dict_from_config(st_dict)
        model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
        model_dict["ordinal_connect"] = st_dict.get("ordinal_connect")
        # model_dict["scaler_weights"] = st_dict.get("scaler_weights")
        return model_dict

    @classmethod
    def get_save_dict(cls, model, optm=None):
        st_dict = super().get_save_dict(model, optm)
        st_dict["n_energy_bins"] = model.n_energy_bins
        st_dict["ordinal_connect"] = model.last_conv.ordinal_connect
        # st_dict["scaler_weights"] = model.scaler.state_dict()
        return st_dict


class PretrainedWatershedUNet(PretrainedUNet):
    def __init__(self, n_energy_bins: int, ordinal_connect: bool, **kwargs):
        super().__init__(**kwargs)

        self.n_energy_bins = n_energy_bins
        self.last_conv = AttachConv(self.n_energy_bins, self.last_conv, ordinal_connect=ordinal_connect)

    @classmethod
    def get_model_dict_from_config(cls, st_dict):
        model_dict = super().get_model_dict_from_config(st_dict)
        model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
        model_dict["ordinal_connect"] = st_dict.get("ordinal_connect")
        return model_dict

    @classmethod
    def get_save_dict(cls, model, optm=None):
        st_dict = super().get_save_dict(model, optm)
        st_dict["n_energy_bins"] = model.n_energy_bins
        st_dict["ordinal_connect"] = model.last_conv.ordinal_connect
        return st_dict


# class WatershedMANetWithScalar(MANetWithScalar):
#     def __init__(self, n_energy_bins: int, ordinal_connect: bool, **kwargs):
#         super().__init__(**kwargs)
#
#         self.n_energy_bins = n_energy_bins
#         self.last_conv = AttachConv(self.n_energy_bins, self.last_conv, ordinal_connect=ordinal_connect)
#
#     @classmethod
#     def get_model_dict_from_config(cls, st_dict):
#         model_dict = super().get_model_dict_from_config(st_dict)
#         model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
#         model_dict["ordinal_connect"] = st_dict.get("ordinal_connect")
#         return model_dict
#
#     @classmethod
#     def get_save_dict(cls, model, optm=None):
#         st_dict = super().get_save_dict(model, optm)
#         st_dict["n_energy_bins"] = model.n_energy_bins
#         st_dict["ordinal_connect"] = model.last_conv.ordinal_connect
#         return st_dict


class WatershedUNetWithScalar(UNetWithScalar):
    def __init__(self, n_energy_bins: int, ordinal_connect: bool, **kwargs):
        super().__init__(**kwargs)

        self.n_energy_bins = n_energy_bins
        self.last_conv = AttachConv(self.n_energy_bins, self.last_conv, ordinal_connect=ordinal_connect)

    @classmethod
    def get_model_dict_from_config(cls, st_dict):
        model_dict = super().get_model_dict_from_config(st_dict)
        model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
        model_dict["ordinal_connect"] = st_dict.get("ordinal_connect")
        return model_dict

    @classmethod
    def get_save_dict(cls, model, optm=None):
        st_dict = super().get_save_dict(model, optm)
        st_dict["n_energy_bins"] = model.n_energy_bins
        st_dict["ordinal_connect"] = model.last_conv.ordinal_connect
        return st_dict

# class WatershedUNetWithScalarV2(UNetWithScalarV2):
#     def __init__(self, n_energy_bins: int, ordinal_connect: bool, **kwargs):
#         super().__init__(**kwargs)
#
#         self.n_energy_bins = n_energy_bins
#         self.last_conv = AttachConv(self.n_energy_bins, self.last_conv, ordinal_connect=ordinal_connect)
#
#     @classmethod
#     def get_model_dict_from_config(cls, st_dict):
#         model_dict = super().get_model_dict_from_config(st_dict)
#         model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
#         model_dict["ordinal_connect"] = st_dict.get("ordinal_connect")
#         return model_dict
#
#     @classmethod
#     def get_save_dict(cls, model, optm=None):
#         st_dict = super().get_save_dict(model, optm)
#         st_dict["n_energy_bins"] = model.n_energy_bins
#         st_dict["ordinal_connect"] = model.last_conv.ordinal_connect
#         return st_dict


# class WatershedUNetWithGeofuse(UNetWithGeofuse):
#     def __init__(self, n_energy_bins: int, ordinal_connect: bool, **kwargs):
#         super().__init__(**kwargs)
#
#         self.n_energy_bins = n_energy_bins
#         self.last_conv = AttachConv(self.n_energy_bins, self.last_conv, ordinal_connect=ordinal_connect)
#
#     @classmethod
#     def get_model_dict_from_config(cls, st_dict):
#         model_dict = super().get_model_dict_from_config(st_dict)
#         model_dict["n_energy_bins"] = st_dict.get("n_energy_bins")
#         model_dict["ordinal_connect"] = st_dict.get("ordinal_connect")
#         return model_dict
#
#     @classmethod
#     def get_save_dict(cls, model, optm=None):
#         st_dict = super().get_save_dict(model, optm)
#         st_dict["n_energy_bins"] = model.n_energy_bins
#         st_dict["ordinal_connect"] = model.last_conv.ordinal_connect
#         return st_dict

