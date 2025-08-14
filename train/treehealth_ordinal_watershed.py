import functools
import itertools
import os
import typing
from datetime import datetime
from pathlib import Path

import distmap
import torchvision
from kornia.filters import spatial_gradient
from scipy.ndimage import distance_transform_edt
import numpy as np
import rasterio
import torch
import kornia
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from src.data.base_dataset import get_dataloader
from src.data.collate import default_collate_with_shapely_support

# if kornia.__version__ != '0.7.0':
#     raise NotImplementedError("Update kornia to version 0.7.0")

from kornia.augmentation import RandomGaussianBlur, RandomAffine, RandomRotation, RandomPerspective, \
    AugmentationSequential, RandomSharpness, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
from kornia.geometry.transform import build_pyramid

from config.treehealth import TreeHealthSegmentationConfig
from src.data.watershed_dataset import calculate_size_weights, calculate_edge_weights, WatershedDataset
from src.modelling import helper
from train.base import init_config
from train.ordinal_watershed import OrdinalWatershedTrainer
from train.watershed import WatershedAugmentation
from skimage import filters

from torchvision import transforms


def watershed_mask2(seg_mask: object, out_shape: object) -> object:
    fe = np.zeros(out_shape)
    # We rasterize each polygon independently to ensure a clean edge.
    for i in seg_mask.unique():
        new_mask = seg_mask.clone().detach()
        new_mask[new_mask != i] = 0
        new_mask[new_mask != 0] = 1
        e = distance_transform_edt(new_mask.cpu())
        fe = np.maximum(fe, e)

    sobelx64f = filters.sobel_h(fe)
    sobely64f = filters.sobel_v(fe)
    n = np.sqrt(sobelx64f * sobelx64f + sobely64f * sobely64f)
    n[n == 0] = 1
    sx = (sobelx64f / n)
    sy = (sobely64f / n)
    return np.ceil(fe), sx, sy


class THOWatershedTrainer(OrdinalWatershedTrainer):
    @staticmethod
    def init_aug_transform(config):
        return THOWatershedAugmentation(config)


class THOWatershedAugmentation(WatershedAugmentation):
    def __init__(self, config):
        WatershedAugmentation.__init__(self, config)

        # Deactivate some augmentations in augment_() in segmentation.py
        self.channel_drop_chance = 0
        self.hflip_chance = 0
        self.vflip_chance = 0
        self.pixel_noise_chance = 0
        self.channel_noise_chance = 0
        self.pixel_drop_chance = 0
        self.pixel_drop_p = 0

        # Activate original affine in segmentation.py
        # self.rotate_params_aug = torch.nn.Parameter(torch.tensor((-90.0, 90.0)))
        # self.shear_params_aug = torch.nn.Parameter(torch.tensor(((-25.0, 25.0), (-25.0, 25.0))))

        self.auto_resample = self.base_config.auto_resample  # For validation

        if isinstance(self.base_config.in_channels, list):
            self.channel_count = np.array(self.base_config.in_channels).sum()
        else:
            self.channel_count = self.base_config.in_channels

        self.nir_drop_chance = self.base_config.nir_drop_chance if self.channel_count >= 4 else 0

        self.normalize = self.base_config.normalize
        # These values are based on RGBI images in denmark, spain, germany, and california...
        self.means = [0.407, 0.416, 0.377, 0.502]  # in the sequence of RGBI
        self.stds = [0.144, 0.123, 0.106, 0.140]

        self.normalize_by_dataset = self.base_config.normalize_by_dataset
        self.mean_map = {
            'germany20cm_2022': [0.4281, 0.482, 0.4505, 0.6711],  # in the sequence of RGBI
            'california60cm_2020': [0.4158, 0.4169, 0.3691, 0.498],
            'swiss25cm_2022': [0.4323, 0.5509, 0.5452, 0.525],
            'spain25cm_2022': [0.4861, 0.4531, 0.402, 0.],
            'finland40cm_2021': [0.3661, 0.3777, 0.3671, 0.3235],
            'denmark20cm_2020': [0.3543, 0.3946, 0.3799, 0.4767],
        }
        self.std_map = {
            'germany20cm_2022': [0.2006, 0.1738, 0.1549, 0.1651],
            'california60cm_2020': [0.1488, 0.1253, 0.1043, 0.1492],
            'swiss25cm_2022': [0.2235, 0.2173, 0.1597, 0.2564],
            'spain25cm_2022': [0.1605, 0.1402, 0.1373, 1e-6],
            'finland40cm_2021': [0.1684, 0.1604, 0.1474, 0.1396],
            'denmark20cm_2020': [0.1323, 0.1121, 0.0986, 0.1293],
        }

        self.mean_imagenet = [0.485, 0.456, 0.406, self.means[-1]]
        self.std_imagenet = [0.229, 0.224, 0.225, self.stds[-1]]

        # Turn this on when epoch is equal to 1... which can save all sample, train, val, and test
        self.save_augmented_patch = True if self.base_config.epochs == 1 and self.base_config.processed_dir is not None and self.training else False  # This is only for one epoch...
        self.save_normalized_patch = True if self.base_config.epochs == 1 and self.base_config.processed_dir is not None and self.training else False  # This can be done in one go

        self.scale_effi = 100  # the same as the on used in watershed_dataset.py
        self.resample_list = [0.2, 0.3, 0.4, 0.5, 0.6]  # the list of resolutions in meter to be resampled to, for validation set
        self.resample_all = True  # resample all patches to the above list of resolutions regardless "faking" high-resolution or not, i.e. from 60cm to 20cm
        self.res_tolerance = 0.02  # sometimes the resolution can be 1 or 2 cm off...
        self.save_resampled_patch = True if self.base_config.epochs == 1 and self.base_config.processed_dir is not None else False

    def batch_to_raster(self, x, dataset_name='patch', out_dir='pts_raster', overwrite=False, crs=None, transform=None):
        patch_id, pt = x[:2]
        if len(x) > 2:
            crs, transform = x[2:]
        # patch_id = patch_id.cpu().numpy()
        # pt = pt.cpu().numpy()
        base_dir = Path(self.base_config.processed_dir) / out_dir
        base_dir.mkdir(exist_ok=True)
        cnt_new = '{}_{}'.format(dataset_name, patch_id)
        output_fp = base_dir / f"{cnt_new}.tif"
        ndt = None

        if not overwrite and os.path.exists(output_fp):
            return

        with rasterio.open(output_fp,
                           mode='w',
                           driver='GTiff',
                           height=pt.shape[2],
                           width=pt.shape[1],
                           count=pt.shape[0],
                           dtype='float32',
                           tiled=False,
                           interleave='pixel',
                           nodata=ndt,
                           crs=crs,
                           transform=transform,
                           ) as dst:
            dst.write(pt)

    def recalculate_weights(self, energy_layer, mask, device, calculate_cw=True, calculate_ew=False):
        """recalculate weights for the rescaled patches"""
        if calculate_cw:
            cw = calculate_size_weights(mask, small_object_threshold=self.base_config.small_object_threshold, device=device)
            return cw
        if calculate_ew:
            ew = calculate_edge_weights(mask, energy_layer,
                                        edge_weight=self.base_config.edge_weight,
                                        edge_threshold_pixel=self.base_config.edge_threshold_pixel,
                                        edge_threshold_percent=self.base_config.edge_threshold_percent, device=device)
            return ew

        else:
            return None

    def recalculate_en(self, idx, segmask, device):
        """recalculate energy band for the rescaled patches"""
        en, sx, sy = watershed_mask2(segmask[idx].squeeze(),
                                     (self.base_config.patch_size, self.base_config.patch_size))
        sobel = torch.stack([torch.from_numpy(sx), torch.from_numpy(sy)], dim=0).to(device)
        return sobel, torch.unsqueeze(torch.from_numpy(en).long().to(device), 0)

    def augment_(self, x, ys: list[torch.Tensor], reference_size, device, scale_factors=None, training=True,
                 manual_resample=False, dataset_name='patch'):
        x = x.clone().detach()

        if not training and not self.auto_resample:
            ha, ori_res, patch_ids, ycount, count_weights, edge_weights = ys[-2], ys[-1], ys[-3], ys[-4], ys[-6], ys[-5]
            y_out = [y.clone().detach() for y in ys[:5]]
            y_out.append(torch.tensor(count_weights, dtype=torch.float32))
            y_out.append(torch.tensor(edge_weights, dtype=torch.float32))
            y_out.append(torch.tensor(ycount, dtype=torch.float32))
            y_out.append(torch.tensor(ha, dtype=torch.float32))
            y_out.append(torch.tensor(ori_res, dtype=torch.float32))
            y_out.append(torch.tensor(patch_ids, dtype=torch.float32))
            y_out.append(torch.tensor([1.] * len(patch_ids), dtype=torch.float32))
            return x, y_out

        if not training and self.auto_resample:  # For evaluation set...
            ha, ori_res, patch_ids, ycount, count_weights, edge_weights = ys[-2], ys[-1], ys[-3], ys[-4], ys[-6], ys[-5]
            ys = [y.clone().detach() for y in ys[:5]]
            ydtype = [y.dtype for y in ys]
            # Attach resampled ones to original evaluation batch (20, 30, 40, 50, 60)
            ori_res = torch.round(ori_res * 100) / 100
            unique_res = torch.unique(torch.round(ori_res * 100) / 100)

            if self.resample_all:
                process_res = unique_res
            else:
                process_res = torch.tensor([i for i in unique_res if abs(i - 0.6) < self.res_tolerance])

            if len(process_res) == 0:
                ys.append(torch.tensor(count_weights, dtype=torch.float32))
                ys.append(torch.tensor(edge_weights, dtype=torch.float32))
                ys.append(torch.tensor(ycount, dtype=torch.float32))
                ys.append(torch.tensor(ha, dtype=torch.float32))
                ys.append(torch.tensor(ori_res, dtype=torch.float32))
                ys.append(torch.tensor(patch_ids, dtype=torch.float32))
                ys.append(torch.tensor([1.] * len(patch_ids), dtype=torch.float32))
                return x, ys

            # Identify the patches that need resampling
            idxs = [torch.tensor([i for i, tupl in enumerate(ori_res) if tupl == j]) for j in process_res]
            has_ = [torch.tensor(ha[j]) for j in idxs]
            oris_ = [torch.tensor(ori_res[j]) for j in idxs]
            ycounts_ = [torch.tensor(ycount[j]) for j in idxs]

            to_res = [[i for i in self.resample_list if i > j] for j in process_res] if not self.resample_all else [[i for i in self.resample_list if abs(i-j) > self.res_tolerance] for j in process_res]
            scales_ = [torch.round(torch.tensor([j / t if abs(j-t) > self.res_tolerance else 1. for t in i]) * 100) / 100 for i, j in zip(to_res, process_res)]
            unique_scales = torch.unique(torch.cat(scales_))
            ys = [y.to(dtype=torch.float32) for y in ys]

            # resampling
            def f(m):
                pi_list, scl = m
                ys_p = [ite[pi_list] for ite in ys]
                aug_func = AugmentationSequential(
                    RandomAffine(degrees=0.0, scale=torch.tensor([scl, scl]), p=1.0),
                    same_on_batch=True,
                    data_keys=['input', *['mask'] * len(ys_p)],
                )
                out = aug_func(x[pi_list], *ys_p)
                return out

            params_all = [(torch.tensor([a, h, r, c]), b) for j in range(len(idxs)) for (a, h, r, c), b in
                          itertools.product(zip(idxs[j], has_[j], oris_[j], ycounts_[j]), scales_[j])]
            params = [[i[0].item() for i, j in params_all if j == j0] for j0 in unique_scales]
            out_list = list(map(f, list(zip(params, unique_scales))))
            x_out = torch.cat([i[0] for i in out_list])
            y_out = [torch.cat([y[j] for y in [i[1:] for i in out_list]]) for j in range(len(ys))]

            # Recalculate count for the upper sampled ones
            ycount2 = torch.tensor([len([j for j in torch.unique(y_out[0][i] * y_out[4][i]) if j!=0]) for i in range(y_out[0].shape[0])])
            idx_scale = [j for i in zip(params, unique_scales.numpy()) for j in itertools.product(i[0], [i[1]])]
            # get params_idx based on idx_scale
            params_all_reorder_idx = [idx for k in idx_scale for idx, i in enumerate(params_all) if i[0][0] == k[0] and i[-1] == k[1]]
            # replace the second values in params_all with ycount2
            for (params_idx, yc2) in zip(params_all_reorder_idx, ycount2):
                params_all[params_idx][0][3] = yc2

            # Recalculate energy bands for the resampled patches
            rs = list(map(functools.partial(self.recalculate_en, segmask=y_out[0], device=device), range(y_out[0].shape[0])))
            # replace augmented en, sx, sy to the original ones
            y_out[1] = torch.stack([i[0] for i in rs]).to(torch.float32)
            y_out[2] = torch.stack([i[1] for i in rs]).to(torch.float32)
            y_out = [y.to(dtype=yt, device=device) for y, yt in zip(y_out, ydtype)]
            del rs

            # Recalculate weights for the resampled patches
            count_weights_rescale, edge_weights_rescale = torch.ones_like(y_out[0]), torch.ones_like(y_out[0])
            if self.base_config.apply_count_weights:
                weights_new = self.recalculate_weights(energy_layer=y_out[2], mask=y_out[0], calculate_cw=True, calculate_ew=False, device=device)
                count_weights_rescale = weights_new.to(torch.float32)
                del weights_new
            if self.base_config.apply_edge_weights:
                weights_new = self.recalculate_weights(energy_layer=y_out[2], mask=y_out[0], calculate_cw=False, calculate_ew=True, device=device)
                edge_weights_rescale = weights_new.to(torch.float32)
                del weights_new

            # other information
            has_all = torch.cat([torch.tensor([i[1].item() for i, j in params_all if j == j0])
                                 for j0 in unique_scales])
            oris_all = torch.cat([torch.tensor([i[2].item() for i, j in params_all if j == j0])
                                  for j0 in unique_scales])
            ycounts_all = torch.cat([torch.tensor([i[3].item() for i, j in params_all if j == j0])
                                     for j0 in unique_scales])
            scales_all = torch.cat([torch.tensor([j.item() for i, j in params_all if j == j0])
                                    for j0 in unique_scales])

            ids_all = torch.cat(
                [torch.tensor([i[0].item() for i, j in params_all if j == j0]) for j0 in unique_scales]).to(torch.int8)
            new_res = torch.tensor([b / a for a, b in zip(scales_all, oris_all)], dtype=torch.float32)
            patch_ids_all = torch.tensor([patch_ids[a.item()].item() for i, a in enumerate(ids_all.to(torch.int8))])

            # save resampled patches
            if self.save_resampled_patch:
                fn_names = ['_'.join([str(pi.item()), str(int(res.item() * 100))]) for pi, res in
                           zip(patch_ids_all, new_res)]
                batch_out = []
                batch_out.append(x_out)
                batch_out.extend([y_out[i] for i in range(len(y_out))])
                batch_out.extend([count_weights_rescale]) if self.base_config.apply_count_weights else None
                batch_out.extend([edge_weights_rescale]) if self.base_config.apply_edge_weights else None
                batch_out = torch.cat(batch_out, 1)
                list(map(functools.partial(self.batch_to_raster, dataset_name=dataset_name, out_dir='resampled_pts',
                                           overwrite=True),
                         list(zip(fn_names, batch_out.cpu()))))

            # append the unprocessed patches
            x_out = torch.cat([x_out, x]).to(torch.float32)
            y_out = [torch.cat([y_out[j], ys[j]]) for j in range(len(ys))]
            y_out = [y.to(dtype=yt, device=device) for y, yt in zip(y_out, ydtype)]

            count_weights_all = torch.cat([count_weights_rescale, count_weights])
            edge_weights_all = torch.cat([edge_weights_rescale, edge_weights])
            ycounts_all = torch.cat([ycounts_all, ycount])
            has_all = torch.cat([has_all, ha])
            patch_res_all = torch.cat([new_res, ori_res])
            patch_ids_all = torch.cat([patch_ids_all, patch_ids])
            scales_all = torch.cat([scales_all, torch.tensor([1.] * x.shape[0])])

            y_out.append(torch.tensor(count_weights_all, dtype=torch.float32))
            y_out.append(torch.tensor(edge_weights_all, dtype=torch.float32))
            y_out.append(torch.tensor(ycounts_all, dtype=torch.float32))
            y_out.append(torch.tensor(has_all, dtype=torch.float32))
            y_out.append(torch.tensor(patch_res_all, dtype=torch.float32))
            y_out.append(torch.tensor(patch_ids_all, dtype=torch.float32))
            y_out.append(torch.tensor(scales_all, dtype=torch.float32))

            return x_out, y_out

        if training:
            ha, ori_res, patch_ids, ycount = ys[-2], ys[-1], ys[-3], ys[-4]
            ys = [y.clone().detach() for y in ys[:7]]
            ydtype = [y.dtype for y in ys]

            # x, ys = super().augment_(x, ys, reference_size, device)  # add noises

            # drop NIR band randomly
            if self.nir_drop_chance != 0 and self.channel_count >= 4:
                drop_chance = torch.floor(torch.rand((self.base_config.batch_size, 1, 1, 1), device=device) + (1 - self.nir_drop_chance))
                keep_ = torch.full((self.base_config.batch_size, self.channel_count - 1, 1, 1), 1, device=device)
                x *= torch.cat((keep_, drop_chance), 1)

            # Basic geometric transformation for both x and ys...
            aug_list_xy = AugmentationSequential(
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomAffine(degrees=0.05, shear=(-25, 25), p=0.5),
                RandomRotation([-90, 90], p=0.5),
                RandomPerspective(distortion_scale=0.01, p=0.2),

                RandomGaussianBlur((3, 3), (0.1, 1.0), p=0.3),
                RandomSharpness(sharpness=0.5, p=0.5),

                same_on_batch=False,
                data_keys=['input', *['mask']*len(ys)],
            )

            ys = [y.to(dtype=torch.float32) for y in ys]
            after_imgaug = aug_list_xy(x, *ys)
            x, ys = after_imgaug[0], after_imgaug[1:]

            patch_res = ori_res.clone().detach()
            scales_out = torch.ones_like(ori_res)

            if scale_factors is not None:
                ys = [y.to(dtype=torch.float32) for y in ys]
                scale_factors = [(i / self.scale_effi, j / self.scale_effi) for i, j in scale_factors]  # the number multiply factor 10 needs to be adjusted according to how the scale_factors were manupulated when creating the patches in the dataset script...
                unique_sf = list(sorted(set(scale_factors)))
                if unique_sf != [(1., 1.)]:
                    process_sf = [i for i in unique_sf if i != (1., 1.)]
                    idxs = [[i for i, tupl in enumerate(scale_factors) if tupl == j] for j in process_sf] # indices of the patches in a batch that need "rescale" augmentation

                    for id_, sf in enumerate(process_sf):
                        if sf != [1., 1.]:
                            ys_p = [ys[j][idxs[id_]] for j in range(len(ys))]
                            x_p = x[idxs[id_]]

                            aug_list_xy = AugmentationSequential(
                                RandomAffine(degrees=0, scale=list(sf), p=0.5),
                                same_on_batch=False,
                                data_keys=['input', *['mask'] * len(ys_p)],
                            )

                            after_imgaug = aug_list_xy(x_p, *ys_p)
                            x_p, ys_p = after_imgaug[0], after_imgaug[1:]

                            scales_ = torch.tensor([1] * len(idxs[id_]), dtype=torch.float32)
                            alt_idx = [i for i, _ in enumerate(aug_list_xy._params[0].data["batch_prob"]) if _ != 0]
                            scales_[alt_idx] = torch.tensor([i[0] for i in aug_list_xy._params[0].data["scale"]], dtype=torch.float32)

                            patch_res[idxs[id_]] = torch.tensor([b / a for a, b in zip(scales_, ori_res[idxs[id_]])], dtype=torch.float32)
                            scales_out[idxs[id_]] = scales_

                            # Remove image augmentation on the y bands, aka recalculate energy bands for resampled patches
                            rs = list(map(functools.partial(self.recalculate_en, segmask=ys_p[0], device=device),
                                          list(range(ys_p[0].shape[0]))))

                            ys_0 = ys_p[0].clone().detach()
                            ys_0[ys_0 != 0] = 1
                            # Replace to the augmented patches
                            x[idxs[id_]] = x_p
                            ys[0][idxs[id_]] = ys_0
                            ys[1][idxs[id_]] = torch.stack([i[0] for i in rs]).to(torch.float32)
                            ys[2][idxs[id_]] = torch.stack([i[1] for i in rs]).to(torch.float32)
                            ys[3][idxs[id_]] = ys_p[3]
                            ys[4][idxs[id_]] = ys_p[4]

                            # Recalculate weights for the rescaled patches
                            if self.base_config.apply_count_weights:
                                weights_new = self.recalculate_weights(energy_layer=ys[2], mask=ys[0], calculate_cw=True, calculate_ew=False, device=device)
                                ys[5] = weights_new.to(torch.float32)
                                del weights_new
                            if self.base_config.apply_edge_weights:
                                weights_new = self.recalculate_weights(energy_layer=ys[2], mask=ys[0], calculate_cw=False, calculate_ew=True, device=device)
                                ys[6] = weights_new.to(torch.float32)
                                del weights_new

                            if self.save_augmented_patch:
                                batch_out = []
                                batch_out.append(x_p)
                                batch_out.extend([ys[i][idxs[id_]] for i in range(len(ys))])
                                batch_out = torch.cat(batch_out, 1)
                                list(map(functools.partial(self.batch_to_raster, dataset_name=dataset_name, out_dir='augmented_pts', overwrite=True), list(zip(patch_ids[idxs[id_]].cpu(), batch_out.cpu()))))
                            del ys_0, rs, x_p

            ys = [y.to(dtype=yt, device=device) for y, yt in zip(ys, ydtype)]
            ys.append(torch.tensor(ha, dtype=torch.float32))
            ys.append(torch.tensor(patch_res, dtype=torch.float32))
            ys.append(torch.tensor(scales_out, dtype=torch.float32))

            torch.cuda.empty_cache()
            return x, ys

    # def get_input(self, sample, device): # prediction and training
    #     # we know that the images are scaled between 0 - 255
    #     # m = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')
    #     return super().get_input(sample, device) / 255.

    # Attach dinov2 bands to the raw images...
    def get_input(self, sample, device, dataset_name='patch'):
        img_keys = self.base_config.image_srcs.keys()

        if self.base_config.band_sequence is not None:
            sample[self.base_config.reference_source] = sample[self.base_config.reference_source][:, self.base_config.band_sequence[:self.channel_count], :, :]

        if len(img_keys) == 1 or self.base_config.append_data is False:
            x0 = super().get_input(sample, device)
            if self.base_config.max_value != 0:
                x = x0 / self.base_config.max_value
            else:
                x = x0.clone().detach()

            # Normalization based on global stats
            if self.normalize:
                if self.normalize_by_dataset:
                    assert len([i for i in sample['src_name'] if i not in self.mean_map.keys()]) == 0, \
                        f"Std not found for {sample['src_name']}"
                    means = torch.tensor([self.mean_map.get(s, None) for s in sample['src_name']], dtype=torch.float32)
                    stds = torch.tensor([self.std_map.get(s, None) for s in sample['src_name']], dtype=torch.float32)
                    x = torch.reshape(torch.cat(tuple(map(lambda i: transforms.Normalize(means[i][:self.channel_count],
                                                                          stds[i][:self.channel_count])(x[i]),
                                            range(x.shape[0])))), x.shape)
                else:
                    if self.base_config.normalize_by_imagenet:
                        means, stds = self.mean_imagenet, self.std_imagenet
                    else:
                        means, stds = self.means, self.stds
                    x = transforms.Normalize(means[:self.channel_count], stds[:self.channel_count])(x)

                if self.save_normalized_patch:
                    list(map(functools.partial(self.batch_to_raster, dataset_name=dataset_name,
                                               out_dir='normalized_pts', overwrite=False),
                             list(zip(sample["patch_id"].cpu(), x.cpu(), sample['ori_crs']['init'],
                                      sample['ori_transform']))))

            # After
            torch.cuda.empty_cache()
            # x = (x - self.means) / self.stds
            # x[x0 == 0.] = np.nan # WHY??? remove nan values
            return x

        else:
            if 'dinov2_features' not in img_keys:
                ref_in_channels = [True] * 4
                x = sample[self.reference_source][:, ref_in_channels] / self.base_config.max_value
                x = torch.cat([x].extend([sample[i] for i in img_keys if i != self.reference_source]), dim=1).to(device)
            else:
                ref_in_channels = [True] * 4
                x = sample[self.reference_source][:, ref_in_channels] / self.base_config.max_value
                dinov2_features = sample['dinov2_features'] / 10000
                x = torch.cat([x, dinov2_features], dim=1).to(device)

            x_m = ~torch.isnan(x)
            x = torch.where(x_m, x, torch.tensor(0.0).to(device))

            save_as_raster = False
            if save_as_raster:
                base_dir = os.path.join(self.base_config.processed_dir, 'extracted_pts_dinov2')
                os.makedirs(base_dir, exist_ok=True)

                for i, idx in enumerate(sample['patch_id']):
                    cnt_new = 'patch_{}'.format(idx)
                    output_fp = os.path.join(base_dir, f"{cnt_new}.tif")
                    ndt = None

                    if not os.path.exists(output_fp):
                        rst = x[i, :, :, :].cpu()
                        with rasterio.open(output_fp,
                                           mode='w',
                                           driver='GTiff',
                                           height=rst.shape[1],
                                           width=rst.shape[2],
                                           count=rst.shape[0],
                                           dtype='float32',
                                           crs=sample["ori_crs"]['init'][i],
                                           transform=sample["ori_transform"][i],
                                           tiled=False,
                                           interleave='pixel',
                                           nodata=ndt
                                           ) as dst:
                            dst.write(rst)

            # After
            torch.cuda.empty_cache()
            return x