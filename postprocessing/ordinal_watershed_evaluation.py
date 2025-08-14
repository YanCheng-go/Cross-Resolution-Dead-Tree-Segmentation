import functools
import itertools
import logging
import shutil
from copy import copy

import rasterio
import torchmetrics
from affine import Affine
from numpy import ndarray
from pyproj import CRS
from skimage.measure import regionprops_table

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import shapes
import torch
from shapely.geometry import Polygon
from torch.nn.functional import one_hot
from skimage.segmentation import watershed
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from postprocessing.util import batch_to_raster, reproject_raster, copy_nonresampled_predictions
from src.data.base_dataset import get_dataloader
from src.data.collate import default_collate_with_shapely_support
from src.utils.jaccard import jaccard_from_confmat
from train.treehealth_ordinal_watershed_5c import THOWatershedTrainer_5c as THOWatershedTrainer


def remap_indices(tensor):
    """make sure that the indices are consecutive"""
    unique_vals = torch.unique(tensor)
    remap_dict = {val.item(): idx for idx, val in enumerate(unique_vals) if val.item() != 0}
    for val, idx in remap_dict.items():
        tensor[tensor == val] = idx
    return tensor


def calculate_tp_fp(preds, labels, probs, iou_thresh=0.30):
    """Find tp, fp, and scores for each patch.
    Args:
        preds: predicted polygons (rasterized) with a unique ID for each instance
        labels: ground truth polygons (rasterized) with a unique ID for each instance
        probs: probabilities, pixel values before applying sigmoid or softmax
        iou_thresh: iou threshold
        auc_method: method to calculate auc
        pr_list: return precision and recall list
        device: device
        medium_area_range: user-defined size of medium objects, to calculate ap across scales
    Returns:
        average precision
    """

    # Step-1 reset IDs in preds and labels
    preds = remap_indices(preds)
    labels = remap_indices(labels)

    # Step-2 calculate iou per instance
    ious = []
    used_gt_ids = set()
    for det_idx in torch.unique(preds):
        if det_idx == 0:
            continue

        best_iou = 0

        new_mask = (preds == det_idx)
        for gt_idx in torch.unique(labels):
            if gt_idx == 0 or gt_idx in used_gt_ids:
                continue
            new_mask2 = (labels == gt_idx)
            intersection = (new_mask & new_mask2).sum((-2, -1)).float()
            union = (new_mask | new_mask2).sum((-2, -1)).float()
            iou = intersection / union
            iou = torch.where(torch.isnan(iou), torch.zeros_like(iou), iou)
            if iou > best_iou:
                best_iou = iou
                best_gt_id, best_det_id = gt_idx, det_idx

        if best_iou >= iou_thresh:
            ious.append({'det_idx': best_det_id, 'gt_idx': best_gt_id, 'iou': best_iou, 'n_pixels': new_mask2.sum()})
            used_gt_ids.add(best_gt_id)  # Mark this ground truth as used

    # Step-3: Prepare True Positives and False Positives
    tp, fp, fn = torch.zeros_like(torch.unique(preds)), torch.zeros_like(torch.unique(preds)), torch.zeros_like(torch.unique(preds))
    scores = torch.zeros_like(torch.unique(preds)).float()
    n_pixels = torch.zeros_like(torch.unique(preds)).float()
    gt_matched = set()

    # Assign TP or FP to detections
    # Sort the IoUs to match the highest ones first
    ious.sort(key=lambda x: x['iou'], reverse=True)
    for match in ious:
        assert match['gt_idx'] not in gt_matched
        tp[match['det_idx']] = 1
        gt_matched.add(match['gt_idx'])
        scores[match['det_idx']] = probs[preds == match['det_idx']].mean()
        n_pixels[match['det_idx']] = match['n_pixels']

    # False positives for unmatched detections
    for det_idx in torch.unique(preds):
        if det_idx == 0:
            continue
        if det_idx not in [m['det_idx'] for m in ious]:
            fp[det_idx] = 1
            scores[det_idx] = probs[preds == det_idx].mean()
            n_pixels[det_idx] = (preds == det_idx).sum()

    # Remove the first one which is associated with the background id 0
    tp = tp[1:]
    fp = fp[1:]
    scores = scores[1:]
    n_pixels = n_pixels[1:]

    return tp, fp, scores, n_pixels


def calculate_ap(tp, fp, scores, n_labels, auc_method='101point', pr_list=False, device='cuda'):
    """Calculate average precision per patch based on tp, fp, and scores"""

    # Step-4: Calculate precision and recall
    fp_cum = torch.cumsum(fp, dim=0)
    tp_cum = torch.cumsum(tp, dim=0)
    recall = tp_cum / n_labels
    precision = tp_cum / (tp_cum + fp_cum)

    # Rule-1: Avoid division by zero
    precision = torch.where(torch.isnan(precision), torch.tensor(0.0, dtype=precision.dtype).to(device), precision)
    # pr_curve(precision, recall)

    # Step-5: Sort by scores in descending order to calculate precision and recall correctly
    sorted_indices = torch.argsort(-scores)
    # Reorder precision and recall based on sorted_indices
    precision = torch.index_select(precision, 0, sorted_indices)
    recall = torch.index_select(recall, 0, sorted_indices)

    # Rule-2: The precision for recall r is the maximum precision obtained for any recall r' >= r
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = torch.max(precision[i], precision[i + 1])

    # pr_curve(precision, recall)
    # Step-7: Calculate AP using the trapezoidal rule correctly methods in ['trapz', '11point', '101point']
    if auc_method == 'trapz':
        ap50 = torch.trapz(precision, recall)
    elif auc_method == '11point':
        # 11-point interpolation for average precision calculation
        ap50 = 0.0
        for t in torch.linspace(0, 1, 11):  # 11 equally spaced points
            if torch.sum(recall >= t) == 0:
                p = 0
            else:
                p = torch.max(precision[recall >= t])
            ap50 += p / 11
    elif auc_method == '101point':
        # 101-point interpolation for average precision calculation
        ap50 = 0.0
        for t in torch.linspace(0, 1, 101):  # 101 equally spaced points
            if torch.sum(recall >= t) == 0:
                p = 0
            else:
                p = torch.max(precision[recall >= t])
            ap50 += p / 101
    else:
        raise NotImplementedError

    if pr_list:
        return [precision, recall]
    else:
        return torch.tensor(ap50).to(device), torch.sum(tp).to(device)


def bulk_patch_ap(preds, labels, probs, iou_thresh=0.50, auc_method='101point', pr_list=False, device='cuda'):
    """Calculate average precision per patch for a batch of patches."""

    out = list(map(lambda i: patch_ap(preds[i].unsqueeze(0), labels[i].unsqueeze(0), probs[i].unsqueeze(0), iou_thresh,
                                      auc_method, pr_list, device), range(preds.shape[0])))

    ap = [i[0] for i in out]
    tp = [i[1] for i in out]
    return torch.stack(ap), torch.stack(tp)


def patch_ap(preds, labels, probs, iou_thresh=0.50, auc_method='101point', pr_list=False, device='cuda'):
    """Calculate average precision for one patch."""

    tp, fp, scores, n_pixels = calculate_tp_fp(preds, labels, probs, iou_thresh=iou_thresh)
    n_labels = torch.sum((torch.unique(labels) != 0))
    out = calculate_ap(tp, fp, scores, n_labels, auc_method=auc_method, pr_list=pr_list, device=device)

    if pr_list:
        return [out[0], out[1]]
    else:
        return out[0], out[1]


def reclassify_by_area(labels, medium_area_range=(75, 389)):
    """Retrieve indices of small, medium, and large instances based on the area."""

    idx_s, idx_m, idx_l = [], [], []

    for gt_idx in torch.unique(labels):
        if gt_idx == 0:
            continue
        new_mask = torch.tensor((labels == gt_idx))
        area = torch.sum(new_mask)
        if area < medium_area_range[0]:
            idx_s.append(gt_idx.long())
        elif medium_area_range[0] <= area < medium_area_range[1]:
            idx_m.append(gt_idx.long())
        elif area >= medium_area_range[1]:
            idx_l.append(gt_idx.long())

    return idx_s, idx_m, idx_l


def bulk_patch_tp_fp(preds, labels, probs, iou_thresh=0.50, include_zeros=False):
    """Calculate tp, fp, and scores a batch of patches."""

    out = list(map(lambda i: calculate_tp_fp(preds[i].unsqueeze(0), labels[i].unsqueeze(0), probs[i].unsqueeze(0), iou_thresh), range(preds.shape[0])))

    if not include_zeros:
        # exclude patches with no labels and no predictions
        idx_zeros = [(torch.sum(preds[i]) == 0) and (torch.sum(labels[i]) == 0) for i in range(preds.shape[0])]
        out = [out[i] for i in range(len(out)) if not idx_zeros[i]]

    # Fill [] in tp, fp, and scores with [0]
    tp = [i[0] if len(i[0]) > 0 else torch.tensor([0]) for i in out]
    fp = [i[1] if len(i[1]) > 0 else torch.tensor([0]) for i in out]
    scores = [i[2] if len(i[2]) > 0 else torch.tensor([0]) for i in out]
    n_pixels = [i[3] if len(i[3]) > 0 else torch.tensor([0]) for i in out]

    return tp, fp, scores, n_pixels


def calculate_mean_score(pred_score, y_seg, idx):
    mean_score = torch.zeros_like(pred_score[idx])
    for i in torch.unique(y_seg[idx]):
        if i == 0:
            continue
        new_mask = (y_seg[idx] == i)
        mean_score[new_mask] = pred_score[idx][new_mask].mean()
    return mean_score


def binary_iou(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    # Calculate intersection and union
    intersection = torch.logical_and(outputs, labels).sum((-2, -1)).float()
    union = torch.logical_or(outputs, labels).sum((-2, -1)).float()

    # Calculate IoU
    iou = intersection / union  # Avoid division by zero
    iou = torch.where(torch.isnan(iou), torch.zeros_like(iou), iou)

    return iou


def retrieve_props(labels: ndarray, out_fp=None, geotransform: Affine = None, crs: str = None,
                   rep_crs=None) -> pd.DataFrame:
    # labels = label(img, connectivity=2)  # set connectivity to 2 to be consistent to the setup used in prediction
    props = regionprops_table(labels, properties=('label', 'centroid', 'orientation', 'axis_major_length',
                                                  'axis_minor_length', 'area', 'area_convex',
                                                  'equivalent_diameter_area', 'area_filled', 'eccentricity',
                                                  'slice', 'perimeter', 'solidity'))

    df_props = pd.DataFrame(props)

    if len(df_props.iloc[:, 0]) <= 0:
        return None

    if geotransform is not None:
        df_props[['x', 'y']] = df_props.apply(lambda row: geotransform * (row['centroid-1'], row['centroid-0']),
                                              axis=1).tolist()
        gdf_props = gpd.GeoDataFrame(df_props, geometry=gpd.points_from_xy(df_props.x, df_props.y), crs=crs)
        gdf_props.crs = crs
        gdf_props['slice'] = gdf_props['slice'].astype(str)

    else:
        gdf_props = df_props

    if rep_crs is not None:
        gdf_props = gdf_props.to_crs(CRS(rep_crs))

    if out_fp is not None:
        gdf_props.to_file(out_fp, driver='GPKG', layer='segmentation_geometry')

    return gdf_props

def log_scalar2(value, log_name):
    if isinstance(value, torch.Tensor):
        value = value.cpu().item()
    logging.info(f"{log_name}: {value}")
    return {log_name: value}

def log_cmat_stats2(cmat, split_name: str, class_names: list):
    stats = {}
    numel = cmat.sum(1)
    mask = numel > 0
    if mask.sum() == 0:  # nothing to log
        return stats
    tp = torch.diag(cmat)[mask]
    stats[f"tp/{split_name}"] = tp.sum().item()
    fp = (cmat.sum(0)[mask] - tp)
    stats[f"fp/{split_name}"] = fp.sum().item()
    fn = (cmat.sum(1)[mask] - tp)
    stats[f"accuracy/{split_name}"] = (tp.sum() / numel.sum())

    # macro statistics
    miou = jaccard_from_confmat(cmat, num_classes=cmat.shape[0], average="none")[mask]
    stats[f"miou/{split_name}"] = miou.mean().item()

    acc = (tp / numel[mask])
    stats[f"macc/{split_name}"] = acc.mean().item()

    precision = tp / (tp + fp + torch.finfo(torch.float32).eps)
    stats[f"precision/{split_name}"] = precision.mean().item()

    recall = tp / (tp + fn + torch.finfo(torch.float32).eps)
    stats[f"recall/{split_name}"] = recall.mean().item()

    f1 = 2 * ((precision * recall) / (precision + recall + torch.finfo(torch.float32).eps))
    stats[f"f1/{split_name}"] = f1.mean().item()

    # also log metrics per class stats
    if len(class_names) == 1:
        class_names = ["0"] + class_names
    for i, class_name in enumerate(np.array(class_names)[mask.cpu().numpy()]):
        stats[f"iou/{split_name}/{class_name}"] = miou[i]
        stats[f"accuracy/{split_name}/{class_name}"] = acc[i]
        stats[f"tp/{split_name}/{class_name}"] = tp[i]
        stats[f"recall/{split_name}/{class_name}"] = recall[i]
        stats[f"precision/{split_name}/{class_name}"] = precision[i]
        stats[f"f1/{split_name}/{class_name}"] = f1[i]

    # logging and resaving dict to ensure that it is not a torch Tensor anymore
    stats = functools.reduce(lambda a, b: {**a, **b}, [log_scalar2(stats[key], key) for key in stats])
    logging.info(f"confusion_matrix/{split_name}: {cmat}")
    return stats


class OrdinalWatershedPostprocessor(THOWatershedTrainer):
    def __init__(self, config):
        super(OrdinalWatershedPostprocessor, self).__init__(config)
        self.config = config
        self.training = False
        self.medium_area_range = (config.medium_area_range_s, config.medium_area_range_m)
        self.iou_thresh = config.iou_thresh

    @staticmethod
    def retrieve_props_loop(i, pred_energy_cumprod, count_mask, batch, reproject_crs='EPSG:4326'):
        w_mask = watershed(-pred_energy_cumprod[i].cpu().numpy(), connectivity=2, mask=count_mask[i])
        if w_mask.sum() == 0:
            return None, batch["patch_id"].numpy()[i]
        else:
            wmask_props = retrieve_props(w_mask, geotransform=batch['ori_transform'][i],
                                         crs=batch['ori_crs']['init'][i], out_fp=None, rep_crs=reproject_crs)
            wmask_props['patch_id'] = batch["patch_id"].numpy()[i]
            wmask_props['resolution'] = np.round(batch['ori_resolution'][i].numpy(), 2)
            wmask_props['resampled'] = 0
            return wmask_props, batch["patch_id"].numpy()[i]

    @staticmethod
    def rasterize_loop(i, batch, pred_energy_cumprod, dataset_name='patch', out_dir='./tmp/pts_raster', overwrite=False, x=None):
        """Export a batch to a raster to the processed_dir with all bands in the batch"""

        patch_id = batch['patch_id'][i]
        if x is not None:
            pt = torch.cat([x[i], pred_energy_cumprod[i].unsqueeze(0),
                            batch['target'][i].unsqueeze(0),
                            batch['target_mask'][i].unsqueeze(0), batch['energy'][i].unsqueeze(0)], dim=0)
        else:
            pt = torch.cat([pred_energy_cumprod[i].unsqueeze(0), batch['target'][i].unsqueeze(0),
                            batch['target_mask'][i].unsqueeze(0), batch['energy'][i].unsqueeze(0)], dim=0)
        crs, transform = batch['ori_crs']['init'][i], batch['ori_transform'][i]
        batch_to_raster((patch_id, pt), dataset_name=dataset_name, out_dir=out_dir, crs=crs,
                        transform=transform, overwrite=overwrite)

    def calculate_count_bias(self, save_df=True, proj_crs='EPSG:4326'):
        """Retrieve count from each training, evaluation, and testing patches
        proj_crs: to project the centroids to a user defined crs"""

        pred_centroids_list = []
        error_list = []
        out_df_list = []
        non_resampled_list = []

        # data loader
        data, config = self.init_data(self.config)
        train_dataset, val_dataset, test_dataset = self.init_dataset_and_split(config, **data)

        self.aug_transform.eval()  # Need to incorporate resolutions
        self.model.eval()

        data_names = ['train', 'val', 'test']
        dataset_all = [train_dataset, val_dataset, test_dataset]

        data_names_sub = list(config.evaluate_datasets.split(','))
        dataset_all = [dataset_all[i] for i, item in enumerate(data_names) if item in data_names_sub]
        data_names = [data_names[i] for i, item in enumerate(data_names) if item in data_names_sub]

        for idx, data_name in enumerate(data_names):
            count_df_list = []
            data_loader = get_dataloader(dataset_all[idx], config.batch_size_val, config.num_workers, collate_fn=default_collate_with_shapely_support, train=False)

            tp_one, fp_one, scores_one, n_labels_one, idx_zeros, idx_s_one, idx_m_one, idx_l_one, idx_resampled, patch_res_all, spatial_cluster_all, n_pixles_one, scales_all = [], [], [], [], [], [], [], [], [], [], [], [], []

            conf_watershed = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device)
            conf_watershed_cumprod = torchmetrics.ConfusionMatrix(task='binary', num_classes=2).to(self.device)
            lpd = 0  # last_progress_digit for verbose visualization
            with tqdm(total=len(data_loader) * config.batch_size_val, desc=f'Evaluating {data_name} set', unit='img', leave=False) as pbar:
                for idx, batch in enumerate(data_loader):

                    spatial_clusters = batch.get("spatial_clusters", None)

                    with torch.no_grad():
                        if config.auto_resample:
                            x, y_seg, y_sobel, y_energy, _, m, count_weights, edge_weights, ycount, ha, patch_res, idxs, scales = self.aug_transform(batch, self.device, dataset_name=data_name)
                            # get object count

                            y_count = copy(ycount).numpy()
                            yenergy = y_energy.clone()
                            spatial_clusters = [dict(zip(batch.get("patch_id", None).numpy(), spatial_clusters)).get(patch_id) for patch_id in idxs.numpy()]
                        else:
                            x, y_seg, y_sobel, y_energy, _, m, count_weights, edge_weights, ycount, ha, patch_res, idxs, scales = self.aug_transform(batch, self.device, dataset_name=data_name)
                            # get object count
                            y_count = batch.get("count", None).numpy()
                            y_count = copy(ycount).numpy()

                        # # change 0.11 or 0.12 to 0.12 in patch_res
                        # patch_res = torch.where(patch_res * 100 - 10 <= 2, 0.12)
                        # patch_res = torch.where(10 - patch_res * 100 <= 2, 0.12)
                        # patch_res = torch.where(patch_res * 100 - 60 <= 2, 0.6)
                        # patch_res = torch.where(60 - patch_res * 100 <= 2, 0.6)

                        additional_scalars = [patch_res.to(self.device)]
                        # viz_batch(x_batch=x, y_batch=y_energy, batch_id=i)
                        if 'with_scalar' in self.config.model_type:
                            pred_sobel, pred_energy = self.model(x, additional_scalars)
                        else:
                            pred_sobel, pred_energy = self.model(x)

                        # Remove edge effects in the evaluation, when the segmentation polygons are on the edge of the patch
                        y_count2 = np.zeros_like(y_count)
                        self.config.remove_edge_on_the_fly = False
                        self.config.reverse_padding = 10
                        if self.config.remove_edge_on_the_fly:
                            non_zeros = torch.nonzero(m, as_tuple=True)
                            col_start, row_start = non_zeros[1][self.get_indices_from_mask(non_zeros)], \
                            non_zeros[2][self.get_indices_from_mask(non_zeros)]

                            non_zeros2 = torch.unbind(torch.flip(torch.stack(non_zeros, dim=1), [0]), dim=1)
                            col_end, row_end = non_zeros2[1][self.get_indices_from_mask(non_zeros2)], non_zeros2[2][
                                self.get_indices_from_mask(non_zeros2)]

                            reverse_padding = torch.tensor(self.config.reverse_padding * scales, device=self.device,
                                                           dtype=torch.uint8)  # 10 is a magic number

                            m_stride = torch.zeros_like(m)
                            for idd in range(m.shape[0]):
                                m_stride[idd,
                                col_start[idd] + reverse_padding[idd]:col_end[idd] - reverse_padding[idd],
                                row_start[idd] + reverse_padding[idd]:row_end[idd] - reverse_padding[idd]] = 1
                                y_count2[idd] = (
                                            torch.unique(y_seg[idd] * (m[idd] * m_stride[idd]), return_counts=True)[
                                                0] != 0).sum().item()

                            m = m_stride.clone()

                        # energy loss
                        y_energy = torch.clamp(y_energy, max=config.n_energy_bins)
                        y_energy = one_hot(y_energy, config.n_energy_bins + 1)[..., 1:]
                        y_energy = y_energy.flip(-1)
                        y_energy.cumsum_(-1)
                        y_energy = y_energy.flip(-1)
                        y_energy.swapaxes_(1, -1)
                        y_energy.squeeze_(-1)

                        # note that this line was not implemented for watershed model v3
                        pred_energy.masked_fill_(~m.unsqueeze(1), 0)  # ensuring that outside areas are empty
                        pred_energy_cumprod = (pred_energy > 0).float().cumprod(1).sum(1)

                        conf_watershed_cumprod.update(
                            (self.get_masked_tensor(pred_energy_cumprod.unsqueeze(1), m) != 0).flatten(),
                            torch.masked_select(y_seg!=0, m).reshape(1, -1).to(dtype=torch.int32).flatten()
                        )

                        # get score (probability) for each predicted object
                        pred_energy_indicator = (pred_energy > 0).float().cumprod(1)
                        # highest prob of one of the activated energy levels per pixel
                        pred_score = (pred_energy.sigmoid() * pred_energy_indicator).amax(1, keepdim=True).squeeze(1)
                        # calculate the average score per instance which has unique values in y_seg
                        mean_score = torch.stack(list(map(lambda i: calculate_mean_score(pred_score, y_seg, i), range(pred_score.shape[0])))).squeeze(1)

                        pred_mask = (pred_energy_cumprod != 0).cpu().numpy()
                        w_mask_ = [watershed(-pred_energy_cumprod[i].cpu().numpy(), connectivity=2, mask=pred_mask[i]) for i in range(m.shape[0])]
                        count_mask = pred_mask * (m.cpu().numpy())
                        w_mask = (w_mask_ * count_mask)
                        watershed_count = np.array([len(np.unique(w_mask[i])) - 1 for i in range(m.shape[0])])

                        # Calculate binary IoU
                        y_seg_m = y_seg * m

                        pred_energy_m = pred_energy_cumprod * m
                        iou_1 = binary_iou((pred_energy_m != 0).float(), (y_seg_m != 0).float())  # IOU per patch???
                        iou_0 = binary_iou((pred_energy_m == 0).float(), (y_seg_m == 0).float())
                        pred_area = pred_energy_m.sum((-2, -1))
                        label_area = y_seg_m.sum((-2, -1))

                        # Calculate average precision for IOU equals to 0.5
                        mean_score_m = mean_score * m
                        # Average over each patch...
                        ap50, tp50 = bulk_patch_ap(torch.stack([torch.tensor(i).to(y_seg.device) for i in w_mask]), y_seg_m, mean_score_m, iou_thresh=self.iou_thresh, device=self.device)
                        # Assign 1 to patch-level ap50 for patches where the number of labeled and predicted instances are both 0
                        ap50[(y_count == 0) * (watershed_count == 0)] = 1
                        # Calculate tp, fp, and scores for each patch to calculate ap50 considering all instances instead of averaged ap50 at the patch level
                        tp, fp, scores, n_pixels = bulk_patch_tp_fp(torch.stack([torch.tensor(i).to(y_seg.device) for i in w_mask]), y_seg_m, mean_score_m, iou_thresh=self.iou_thresh, include_zeros=True)
                        tp_one.append(tp)
                        fp_one.append(fp)
                        scores_one.append(scores)
                        n_pixles_one.append(n_pixels)
                        n_labels_one.append(torch.tensor(y_count).to(self.device))
                        idx_zeros.append([torch.tensor((y_count[i] == 0) * (watershed_count[i] == 0)).to(self.device) for i in range(m.shape[0])])
                        idx_resampled.append(torch.tensor([1 if scales[i].numpy() != 1 else 0 for i in range(m.shape[0])]).to(self.device))
                        patch_res_all.append(torch.tensor(patch_res).to(self.device))
                        scales_all.append(torch.tensor(scales).to(self.device))
                        spatial_cluster_all.append(spatial_clusters)  # can be counties and biomes or other, should add a function to get this info from associated shapefile
                        # Retrieve size information to calculate metrics across sizes
                        re = list(map(lambda i: reclassify_by_area(y_seg_m[i], medium_area_range=self.medium_area_range), range(m.shape[0])))
                        idx_s, idx_m, idx_l = list(zip(*re))
                        idx_s_one.append(idx_s)
                        idx_m_one.append(idx_m)
                        idx_l_one.append(idx_l)

                        # Export to patches (predictions and raw input) to rasters
                        if not config.auto_resample:
                            batch_ids = [i for patch_id in idxs for i, item in enumerate(batch['patch_id']) if patch_id == item]

                            # Extract centroids
                            out_ = list(map(functools.partial(self.retrieve_props_loop,
                                                              pred_energy_cumprod=pred_energy_cumprod,
                                                              count_mask=count_mask, batch=batch,
                                                              reproject_crs=config.project_crs), range(m.shape[0])))

                            save_raster_folder = 'pts_raster'
                            if config.save_out_rasters:
                                list(map(functools.partial(self.rasterize_loop, x=x.cpu(),
                                                           batch=batch,
                                                           pred_energy_cumprod=pred_energy_cumprod.cpu(),
                                                           dataset_name=data_name,
                                                           out_dir=Path(config.report_folder, save_raster_folder),
                                                           overwrite=True), range(m.shape[0])))

                        else:
                            out_ = []
                            batch_ids = [i for patch_id in idxs for i, item in enumerate(batch['patch_id']) if patch_id == item]

                            for i, batch_id in enumerate(batch_ids):
                                pt = torch.cat([x[i], pred_energy_cumprod[i].unsqueeze(0), y_seg[i].unsqueeze(0), m[i].unsqueeze(0), yenergy[i]], dim=0)
                                crs, transform = batch['ori_crs']['init'][batch_id], batch['ori_transform'][batch_id]

                                save_raster_folder = 'pt_raster_resampled'
                                if config.save_out_rasters:
                                    batch_to_raster(('_'.join([str(int(idxs[i].item())), str(int(np.round(patch_res[i].numpy(),2) * 100))]),
                                                     pt.cpu().numpy()), dataset_name=data_name, out_dir=Path(config.report_folder) / save_raster_folder,
                                                    crs=crs, transform=transform, overwrite=True)

                                # Extract centroids
                                if w_mask[i].sum() == 0:
                                    out_.append((None, batch["patch_id"].numpy()[batch_id]))
                                else:
                                    wmask_props = retrieve_props(w_mask[i], geotransform=batch['ori_transform'][batch_id], crs=batch['ori_crs']['init'][batch_id], out_fp=None, rep_crs=config.project_crs)
                                    wmask_props['patch_id'] = batch["patch_id"].numpy()[batch_id]
                                    wmask_props['resolution'] = np.round(patch_res[i].numpy(),2)
                                    wmask_props['resampled'] = 0 if scales[i].numpy() == 1 else 1
                                    out_.append((wmask_props, batch["patch_id"].numpy()[batch_id]))

                        error_list.append([i[1] for i in out_ if i[0] is None])
                        wmask_props_list = [i[0] for i in out_ if i[0] is not None]
                        if wmask_props_list == []:
                            continue
                        pred_centroids = gpd.GeoDataFrame(pd.concat(wmask_props_list))
                        pred_centroids['Class'] = data_name
                        pred_centroids_list.append(pred_centroids)
                        del pred_centroids, wmask_props_list, out_

                        # Calculate count bias and save to a dataframe
                        src_names = [batch['src_name'][i] for i in batch_ids]

                        count_df = pd.DataFrame(
                            list(zip(idxs.numpy(), patch_res.numpy(), y_count, y_count2, watershed_count, ha.numpy(),
                                     src_names, pred_area.cpu().numpy(), label_area.cpu().numpy(),
                                     iou_1.cpu().numpy(), iou_0.cpu().numpy(), scales.cpu().numpy(), ap50.cpu().numpy())),
                            columns=['Patch ID', 'Patch resolution', 'Label', 'Label2', 'Prediction', 'hectares',
                                     'src_name', 'pred_area', 'label_area', 'iou_1', 'iou_0', 'scales', 'ap50'])
                        count_df["Difference"] = count_df.Prediction - count_df.Label
                        count_df["Difference2"] = count_df.Prediction - count_df.Label2
                        count_df["Class"] = data_name
                        count_df = count_df.sort_values('Patch ID').reset_index(drop=True)
                        count_df['count_loss_ha'] = count_df['Difference'] / count_df['hectares']
                        count_df['count_loss_ha2'] = count_df['Difference2'] / count_df['hectares']
                        count_df['resampled'] = count_df['scales'].apply(lambda x: 0 if x == 1. else 1.)
                        count_df_list.append(count_df)

                        non_resampled_list.extend(['_'.join(i) for i in
                                              zip(count_df[count_df['resampled'] == 0]['Class'].tolist(),
                                                  count_df[count_df['resampled'] == 0]['Patch ID'].apply(
                                                      lambda x: str(int(x))).tolist(),
                                                  count_df[count_df['resampled'] == 0]['Patch resolution'].apply(
                                                      lambda x: str(int(np.round(x, 2) * 100))).tolist())])

                        del count_df
                        pbar.update(x.shape[0])

            count_df = pd.concat(count_df_list)

            # # calculate dataset-level metrics
            # # Count bias (%)
            # count_bias = (count_df['Difference'] * (-1)).sum() / count_df['Label'].sum() * 100
            # logging.info(f"bias(%)/{data_name}: {count_bias}")
            # count_bias_ori = (count_df[count_df['resampled']==0]['Difference'] * (-1)).sum() / count_df[count_df['resampled']==0]['Label'].sum() * 100
            # logging.info(f"bias(%)_ORI/{data_name}: {count_bias_ori}")
            #
            # cmat_watershed_cumprod = conf_watershed_cumprod.compute()
            # out_cmat = log_cmat_stats2(cmat_watershed_cumprod, f"watershed_cumprod/{data_name}", self.class_names)

            # Calculate ap50 metrics for val, test, and train set, not as an average of patch-level metrics
            # 1. Calculate ap50 for non-resampled ones
            # Exclude zero patches and resampled patches (when auto_resample is True)
            if config.auto_resample:
                idx_select_m = [[j for j, _ in enumerate(idx_resampled[i]) if _ == 0 and not idx_zeros[i][j]] for i in range(len(data_loader))]
            else:
                idx_select_m = [[j for j, _ in enumerate(idx_zeros[i]) if not idx_zeros[i][j]] for i in range(len(data_loader))]

            # Cat tp, fp and scores
            tp_one_m = torch.cat([torch.cat([tp_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            fp_one_m = torch.cat([torch.cat([fp_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            scores_one_m = torch.cat([torch.cat([scores_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            n_labels_one_m = torch.cat([n_labels_one[i][[j for j in idx_select_m[i]]] for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            # Calculate metrics
            ap50_one, tp50_one = calculate_ap(tp_one_m, fp_one_m, scores_one_m, n_labels_one_m.sum(), auc_method='101point', pr_list=False, device=self.device)
            logging.info(f"\nIOU = {self.iou_thresh}")
            logging.info(f"AP for {data_name} dataset (non-resampled): {np.round(ap50_one.numpy() ,4)}")
            logging.info(f"TP for {data_name} dataset (non-resampled): {np.round(tp50_one.numpy())}")

            # 2. AP50 across difference size categories only for non-resampled ones
            # Size categories -> user-defined based on statistics, which may be different across datasets, resolution, and species or biomes);
            # Exclude exclude zero patches in n_pixels
            n_pixels_one_m = torch.cat([torch.cat([n_pixles_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            # Get number of labels per size category
            n_labels_s_m = torch.cat([torch.tensor([len(idx_s_one[i][j]) for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            n_labels_m_m = torch.cat([torch.tensor([len(idx_m_one[i][j]) for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
            n_labels_l_m = torch.cat([torch.tensor([len(idx_l_one[i][j]) for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])

            # Calculate metrics per size categories....
            l_mask = (n_pixels_one_m > self.medium_area_range[-1])
            ap50_l_one, tp50_l_one = calculate_ap(tp_one_m * l_mask, fp_one_m * l_mask, scores_one_m * l_mask, n_labels_l_m.sum(), auc_method='101point', pr_list=False, device=self.device)
            logging.info(f"AP for large objects (> {self.medium_area_range[-1]} pixels) in {data_name} dataset (non-resampled): {np.round(ap50_l_one.numpy(), 4)}")
            logging.info(f"TP for large objects (< {self.medium_area_range[-1]} pixels) in {data_name} dataset (non-resampled): {np.round(tp50_l_one.numpy())}")

            m_mask = ((n_pixels_one_m <= self.medium_area_range[-1]) * (n_pixels_one_m > self.medium_area_range[0]))
            ap50_m_one, tp50_m_one = calculate_ap(tp_one_m * m_mask, fp_one_m * m_mask, scores_one_m * m_mask, n_labels_m_m.sum(), auc_method='101point', pr_list=False, device=self.device)
            logging.info(f"AP for medium objects ({self.medium_area_range[0]} - {self.medium_area_range[-1]} pixels) {data_name} dataset (non-resampled): {np.round(ap50_m_one.numpy(), 4)}")
            logging.info(f"TP for medium objects ({self.medium_area_range[0]} - {self.medium_area_range[-1]} pixels) {data_name} dataset (non-resampled): {np.round(tp50_m_one.numpy())}")

            s_mask = (n_pixels_one_m <= self.medium_area_range[0])
            ap50_s_one, tp50_s_one = calculate_ap(tp_one_m * s_mask, fp_one_m * s_mask, scores_one_m * s_mask, n_labels_s_m.sum(), auc_method='101point', pr_list=False, device=self.device)
            logging.info(f"AP for small objects (< {self.medium_area_range[0]} pixels) {data_name} dataset (non-resampled): {np.round(ap50_s_one.numpy(), 4)}")
            logging.info(f"TP for small objects (< {self.medium_area_range[0]} pixels) {data_name} dataset (non-resampled): {np.round(tp50_s_one.numpy())}")

            # Exclude only zero patches
            idx_select = [[j for j, _ in enumerate(idx_zeros[i]) if not idx_zeros[i][j]] for i in range(len(data_loader))]
            # 4. AP50 for resampled and non-resampled ones
            if config.auto_resample:
                # Cat tp, fp and scores
                tp_one_m = torch.cat([torch.cat([tp_one[i][j] for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                fp_one_m = torch.cat([torch.cat([fp_one[i][j] for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                scores_one_m = torch.cat([torch.cat([scores_one[i][j] for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                n_labels_one_m = torch.cat([n_labels_one[i][[j for j in idx_select[i]]] for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                ap50_one, tp50_one = calculate_ap(tp_one_m, fp_one_m, scores_one_m, n_labels_one_m.sum(), auc_method='101point', pr_list=False, device=self.device)
                logging.info(f"\nIOU = {self.iou_thresh}")
                logging.info(f"AP for {data_name} dataset (resampled and non-resampled): {np.round(ap50_one.numpy(), 4)}")
                logging.info(f"TP for {data_name} dataset (resampled and non-resampled): {np.round(tp50_one.numpy())}")

                # Exclude exclude zero patches in n_pixels
                n_pixels_one_m = torch.cat([torch.cat([n_pixles_one[i][j] for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                # Get number of labels per size category
                n_labels_s_m = torch.cat([torch.tensor([len(idx_s_one[i][j]) for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                n_labels_m_m = torch.cat([torch.tensor([len(idx_m_one[i][j]) for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])
                n_labels_l_m = torch.cat([torch.tensor([len(idx_l_one[i][j]) for j in idx_select[i]]) for i in range(len(data_loader)) if len(idx_select[i]) > 0])

                # Calculate metrics per size categories....
                l_mask = (n_pixels_one_m > self.medium_area_range[-1])
                ap50_l_one, tp50_l_one = calculate_ap(tp_one_m * l_mask, fp_one_m * l_mask, scores_one_m * l_mask, n_labels_l_m.sum(), auc_method='101point', pr_list=False, device=self.device)
                logging.info(f"AP for large objects (> {self.medium_area_range[-1]} pixels) in {data_name} dataset (resampled and non-resampled): {np.round(ap50_l_one.numpy(), 4)}")
                logging.info(f"TP for large objects (< {self.medium_area_range[-1]} pixels) in {data_name} dataset (resampled and non-resampled): {np.round(tp50_l_one.numpy())}")

                m_mask = ((n_pixels_one_m <= self.medium_area_range[-1]) * (n_pixels_one_m > self.medium_area_range[0]))
                ap50_m_one, tp50_m_one = calculate_ap(tp_one_m * m_mask, fp_one_m * m_mask, scores_one_m * m_mask, n_labels_m_m.sum(), auc_method='101point', pr_list=False, device=self.device)
                logging.info(f"AP for medium objects ({self.medium_area_range[0]} - {self.medium_area_range[-1]} pixels) {data_name} dataset (resampled and non-resampled): {np.round(ap50_m_one.numpy(), 4)}")
                logging.info(f"TP for medium objects ({self.medium_area_range[0]} - {self.medium_area_range[-1]} pixels) {data_name} dataset (resampled and non-resampled): {np.round(tp50_m_one.numpy())}")

                s_mask = (n_pixels_one_m <= self.medium_area_range[0])
                ap50_s_one, tp50_s_one = calculate_ap(tp_one_m * s_mask, fp_one_m * s_mask, scores_one_m * s_mask, n_labels_s_m.sum(), auc_method='101point', pr_list=False, device=self.device)
                logging.info(f"AP for small objects (< {self.medium_area_range[0]} pixels) {data_name} dataset (resampled and non-resampled): {np.round(ap50_s_one.numpy(), 4)}")
                logging.info(f"TP for small objects (< {self.medium_area_range[0]} pixels) {data_name} dataset (resampled and non-resampled): {np.round(tp50_s_one.numpy())}")

            # 5. AP50 across resolutions and datasets
            patch_res_all_ = torch.cat([patch_res_all[i][[j for j in idx_select[i]]] for i in range(len(data_loader))])
            scales_all_ = torch.cat([scales_all[i][[j for j in idx_select[i]]] for i in range(len(data_loader))])
            spatial_cluster_all_ = np.array(list(itertools.chain(*[[spatial_cluster_all[i][j] for j in idx_select[i]] for i in range(len(data_loader))])))
            for sc, res, scale_ in set(zip(spatial_cluster_all_, patch_res_all_.numpy(), scales_all_.numpy())):
                idx_select_m = [[j for j, _ in enumerate(idx_zeros[i]) if (not idx_zeros[i][j]) and patch_res_all[i][j] == res and spatial_cluster_all[i][j] == sc and scales_all[i][j] == scale_] for i in range(len(data_loader))]
                tp_one_m = torch.cat([torch.cat([tp_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
                fp_one_m = torch.cat([torch.cat([fp_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
                scores_one_m = torch.cat([torch.cat([scores_one[i][j] for j in idx_select_m[i]]) for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
                n_labels_one_m = torch.cat([n_labels_one[i][[j for j in idx_select_m[i]]] for i in range(len(data_loader)) if len(idx_select_m[i]) > 0])
                ap50_one, tp50_one = calculate_ap(tp_one_m, fp_one_m, scores_one_m, n_labels_one_m.sum(), auc_method='101point', pr_list=False, device=self.device)
                logging.info(f"AP for {data_name} in {sc} at resolution {np.round(res,2)} (scale: {scale_}): {np.round(ap50_one.numpy(), 4)}")
                logging.info(f"TP for {data_name} in {sc} at resolution {np.round(res,2)} (scale: {scale_}): {np.round(tp50_one.numpy())}")

            iou_suffix = f'_0p{str(int(self.iou_thresh * 10))}' if self.iou_thresh is not None else ''

            time_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
            if save_df is True:
                count_df.to_csv(os.path.join(config.report_folder, f'{time_prefix}_post-watershed_count-bias_label-vs-prediction_{data_name}{iou_suffix}.csv'), index=False)
                patch_grid = gpd.read_file(os.path.join(config.processed_dir, f"qgis/{data_name}/patch_grid.gpkg")) if config.split_col is not None else gpd.read_file(os.path.join(config.processed_dir, f"qgis/patches/patch_grid.gpkg"))
                out_df = pd.merge(patch_grid, count_df, left_on="patch_id", right_on="Patch ID")
                out_df_list.append(out_df)
                out_fp = os.path.join(config.report_folder, f'{time_prefix}_post-watershed_count-bias_label-vs-prediction_{data_name}{iou_suffix}.gpkg')
                out_df.to_file(out_fp, driver='GPKG', layer='Bias')
                print(f"The table is saved here: {out_fp}")
                del out_df
            del count_df, count_df_list

        # Combine all centroids of all patches into one
        if config.project_crs is not None:
            gpd.GeoDataFrame(pd.concat(pred_centroids_list)).to_file(Path(config.report_folder) / f'{time_prefix}_predictions_centroids{iou_suffix}.gpkg', crs=proj_crs)
            if config.save_out_rasters is True and os.path.exists(Path(config.report_folder) / save_raster_folder):
                reproject_raster(out_dir=Path(config.report_folder) / f'{save_raster_folder}_proj', in_dir=Path(config.report_folder, save_raster_folder), dst_crs=config.project_crs, overwrite=False)
                if self.config.auto_resample is True:
                    in_dir = Path(config.report_folder) / f'{save_raster_folder}_proj'
                    out_dir = Path(config.report_folder) / f'{save_raster_folder}_proj'.replace('_resampled', '')

                    # add 0.11 to 0.12 and 0.12 to 0.11
                    non_resampled_list_1 = [i.replace('12', '11') for i in non_resampled_list]
                    non_resampled_list_2 = [i.replace('11', '12') for i in non_resampled_list]
                    non_resampled_list = list(set(non_resampled_list_1 + non_resampled_list + non_resampled_list_2))

                    copy_nonresampled_predictions(in_dir, out_dir, fn_list=non_resampled_list)

        # Combine count_df for all patches into one
        out_df = pd.concat(out_df_list)
        time_prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
        if save_df is True:
            out_fp = os.path.join(config.report_folder, f'{time_prefix}_post-watershed_count-bias_label-vs-prediction{iou_suffix}.gpkg')
            out_df.to_file(out_fp, driver='GPKG', layer='Bias')
            print(f"The table is saved here: {out_fp}")

        return out_df


if __name__ == '__main__':

    # # continue moving non-resampled predictions
    # in_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v20240626_data20240522_countWeights/pt_raster_resampled_proj'
    # out_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v20240626_data20240522_countWeights/pt_raster_proj'
    # non_resampled_list = []
    # count_df0 = pd.read_csv('/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v20240626_data20240522_countWeights/20240628-142705_post-watershed_count-bias_label-vs-prediction_val.csv')
    # count_df1 = pd.read_csv('/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v20240626_data20240522_countWeights/20240628-144418_post-watershed_count-bias_label-vs-prediction_test.csv')
    # count_df = pd.concat([count_df0, count_df1])
    #
    # # combine three columns into one column in count_df
    # count_df['Patch ID'] = count_df['Patch ID'].apply(lambda x: str(int(x)))
    # count_df['Patch resolution'] = count_df['Patch resolution'].apply(lambda x: str(int(np.round(x, 2) * 100)))
    # count_df['Class'] = count_df['Class'].apply(lambda x: x.split('_')[0])
    # count_df['non_resampled_id'] = count_df['Class'] + '_' + count_df['Patch ID'] + '_' + count_df['Patch resolution']
    # non_resampled_list = [i.replace('12', '11') for i in count_df[count_df['resampled'] == 0]['non_resampled_id']]
    # # non_resampled_list = [i.replace('60', '61') for i in count_df['non_resampled_id']]
    # # copy_nonresampled_predictions(in_dir, out_dir, fn_list=non_resampled_list)
    #
    # print([i for i in non_resampled_list if i + '.tif' not in os.listdir(out_dir)])

    # Continue combining

    # Add iou threshold to all files, modify file names before the update of this script...
    def preprocess_files_in_folder(folder_path):
        # 1. Create an empty dataframe with specified columns
        data = pd.DataFrame(columns=['datetime', 'file_name', 'new_file_name', 'file_path', 'iou_threshold'])

        # 2. Get all files in the folder and save to file_path column
        data['file_path'] = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                             os.path.isfile(os.path.join(folder_path, f))]
        data['file_name'] = data['file_path'].apply(lambda x: os.path.basename(x))

        # 3. Split the base names by '_' and get the first part, save it to datetime column
        data['datetime'] = data['file_name'].apply(lambda x: x.split('_')[0])

        # 4. Make sure the datatype for the datetime column is datetime
        try:
            data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d-%H%M%S')  # Adjust format as needed
        except ValueError as e:
            print(f"Error converting datetime: {e}")

        # 5. Identify rows with file_name ending with specific patterns before the file extension
        iou_pattern = r'0p[3579](?=\.\w+$)'  # regex to match '0p3', '0p7', '0p5', '0p9' just before the file extension
        data['iou_threshold'] = data['file_name'].str.extract(f"({iou_pattern})", expand=False)

        # 6. Filter out rows without a recognized iou_threshold
        data_with_iou = data.dropna(subset=['iou_threshold'], inplace=False)

        # 7. For each of these rows, find 5 closest prior datetime entries
        for index, row in data_with_iou.iterrows():
            # Get all rows with earlier datetimes
            earlier_data = data[data['datetime'] < row['datetime']]
            # Find the 5 closest earlier datetimes
            closest_rows = earlier_data.iloc[(earlier_data['datetime'] - row['datetime']).abs().argsort()[:5]]
            # Update new_file_name in the original data
            for closest_index in closest_rows.index:
                original_path = data.at[closest_index, 'file_path']
                directory, filename = os.path.split(original_path)
                name_part, extension = os.path.splitext(filename)
                new_name = f"{name_part}_{row['iou_threshold']}{extension}"
                new_file_path = os.path.join(directory, new_name)
                data.at[closest_index, 'new_file_name'] = new_file_path

                # Copy the file to the new location with the new name
                shutil.copy(original_path, new_file_path)
        return data

    # # Example usage
    # for i in ['0802', '0803', '0731', '0809', '0808', '0805', '0804', '0728', '0727', '0730', '0725', '0724', '0723', '0721', '0720', '0719']:
    #     folder_path = f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/v2024{i}_data20240801_CH'
    #     result_data = preprocess_files_in_folder(folder_path)
    pass