"""Compare to external evaluation data."""
import os
from pprint import pprint

import geopandas as gpd
import pandas as pd
import rasterio.features
import torchmetrics

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import nan_to_num
from pytz import country_names

from src.utils.jaccard import jaccard_from_confmat

def log_cmat_stats(cmat, split_name: str ='all', class_names: list = ['0', '1']):
    """log confusion matrix statistics"""
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

    # normalize conf matrix
    cmatn = cmat / cmat.sum(axis=1, keepdim=True)
    cmatn = nan_to_num(cmatn.cpu().numpy())
    return stats


def rasterize_polygons(img_fp, geo_fp, bbox=None):
    """rasterize polygons and save in the disk"""
    with rasterio.open(img_fp) as src:
        img_shape = (src.height, src.width)
        img_transform = src.transform
    fr = gpd.read_file(geo_fp).to_crs(src.crs)
    fr = fr['geometry'].to_list()
    fr2 = ((geom, value + 1) for geom, value in zip(fr, list(range(len(fr)))))
    label = rasterio.features.rasterize(fr2, out_shape=img_shape, all_touched=False, fill=0, transform=img_transform)

    if bbox is not None:
        bbox = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[bbox])
        # change the crs for the bbox polygon to the same as the image
        bbox = bbox.to_crs(src.crs)
        bbox = bbox.geometry[0]

        # clip the raster to the bbox
        mask = rasterio.features.geometry_mask([bbox], out_shape=img_shape, transform=img_transform, invert=True)
        label[~mask] = 0

        # get the area of the bbox
        bbox_area = bbox.area

    # # show a three band image
    # with rasterio.open(img_fp) as src:
    #     img = src.read([1, 2, 3])
    #     plt.imshow(img.transpose(1, 2, 0))
    #     plt.imshow(label, alpha=0.5)
    #     plt.show()
    else:
        bbox_area = 0
    return label, bbox_area

def read_predictions(pred_fp, bbox=None):
    """read predictions from disk"""
    with rasterio.open(pred_fp) as src:
        pred = src.read(2)
        img_shape = (src.height, src.width)
        img_transform = src.transform

    if bbox is not None:
        bbox = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[bbox])
        # change the crs for the bbox polygon to the same as the image
        bbox = bbox.to_crs(src.crs)
        bbox = bbox.geometry[0]
        # clip the raster to the bbox
        mask = rasterio.features.geometry_mask([bbox], out_shape=img_shape, transform=img_transform, invert=True)
        pred[~mask] = 0

        # get the area of the bbox
        bbox_area = bbox.area

        # # get the count of each pixel value
        # unique, counts = np.unique(pred, return_counts=True)
        # # mask the value where count is <=2
        # for i in unique[counts <= 2]:
        #     pred[pred == i] = 0

    else:
        bbox_area = 0

    return pred, bbox_area


def calculate_cmat(fps, bboxs=None, df_fp='./tmp/external_evaluation.csv'):
    """calculate confusion matrix"""
    country_list = []
    fn_list = []
    rmae_list = []
    iou_list = []
    f1_list = []
    label_count_list = []
    pred_count_list = []
    bbox_area_list = []

    if bboxs is None:
        bboxs = [None] * len(fps)

    conf_watershed = torchmetrics.ConfusionMatrix(task='binary', num_classes=2)

    for (img_fp, geo_fp, pred_fp), bbox in zip(fps, bboxs):

        county_name = img_fp.split('/')[-6]

        # if not county_name in ['France']:
        #     continue

        label, bbox_area = rasterize_polygons(img_fp, geo_fp, bbox)
        pred, bbox_area = read_predictions(pred_fp, bbox)

        conf_watershed.update(
            torch.from_numpy((pred > 0).flatten()), torch.from_numpy((label > 0).flatten())
        )
        iou = jaccard_from_confmat(conf_watershed.compute(), num_classes=2, average="none")[1]
        iou_list.append(iou.item())
        f1 = 2 * iou / (iou + 1)
        f1_list.append(f1.item())

        pred_count = len(set(pred.flatten()))
        label_count = len(set(label.flatten()))
        rMAE = np.abs(pred_count - label_count) / label_count
        rmae_list.append(rMAE)

        country_list.append(county_name)
        fn_list.append(img_fp)

        label_count_list.append(label_count)
        pred_count_list.append(pred_count)

        bbox_area_list.append(bbox_area)

    df = pd.DataFrame({'country': country_list, 'fn': fn_list, 'bbox_area': bbox_area_list, 'label_count': label_count_list, 'predict_count': pred_count_list, 'iou': iou_list, 'f1': f1_list, 'rmae': rmae_list})
    df.to_csv(df_fp, index=False)

    cmat_watershed = conf_watershed.compute()
    stats = log_cmat_stats(cmat_watershed)

    # calculate bias per country
    print('bias_by_country: ', (df.groupby('country')['label_count'].sum() - df.groupby('country')['predict_count'].sum()) / df.groupby('country')['label_count'].sum() * 100)
    print('bbox_area: ', (df.groupby('country')['bbox_area'].sum()))
    print('label_count: ', (df.groupby('country')['label_count'].sum()))
    print('predict_count: ', (df.groupby('country')['predict_count'].sum()))
    print('bias_all: ', (df['label_count'].sum() - df['predict_count'].sum()) / df['label_count'].sum() * 100)
    return stats, df


def get_fps(counties=['France', 'Poland', 'Estonia', 'Germany', 'Finland', 'Switzerland']):
    """get file paths"""
    with open('/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/deadtrees_images_20241031_to_predict.txt') as f:
        img_fps = f.readlines()
    img_fps = [fp.strip() for fp in img_fps]
    img_fps = [fp for fp in img_fps if any([county in fp for county in counties])]

    geo_fps = [os.path.splitext(fp.replace('Image', 'Geojson'))[0] + '.geojson' for fp in img_fps]

    pred_dir = '/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/predictions/v20241022_p20241104/rasters/'
    pred_fps = [os.path.join(pred_dir, 'det_' + os.path.splitext(os.path.basename(fp))[0] + '.tif') for fp in img_fps]

    return list(zip(img_fps, geo_fps, pred_fps))


if __name__ == '__main__':
    # read bboxs multi polygons
    bboxs = gpd.read_file('/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/external_evaluation/external_evaluation_aois.geojson')
    img_fps = bboxs['location'].to_list()
    # geo_fps = [os.path.splitext(fp.replace('Image', 'Geojson'))[0] + '.geojson' for fp in img_fps]
    geo_fps = ['/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/external_evaluation/external_evaluation_features.geojson'] * len(img_fps)
    pred_fps = [os.path.join('/mnt/raid5/DL_TreeHealth_Aerial/fromSamuli/predictions/v20241022_p20241104/rasters/', 'det_' + os.path.splitext(os.path.basename(fp))[0] + '.tif') for fp in img_fps]
    bboxs = bboxs['geometry'].to_list()

    # for denmark
    bboxs_denmark = gpd.read_file('/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/external_evaluation/external_evaluation_aois_denmark.geojson')
    img_fps_denmark = bboxs_denmark['path'].to_list()
    geo_fps_denmark = ['/mnt/raid5/DL_TreeHealth_Aerial/Merged/training_dataset/external_evaluation/external_evaluation_features_denmark.geojson'] * len(img_fps_denmark)
    pred_fps_denmark = [os.path.join('/mnt/raid5/DL_TreeHealth_Aerial/Denmark/predictions/v20241022_p20241024/rasters/', 'det_' + os.path.splitext(os.path.basename(fp))[0] + '.tif') for fp in img_fps_denmark]
    bboxs_denmark = bboxs_denmark['geometry'].to_list()

    # combine the two
    img_fps = img_fps + img_fps_denmark
    geo_fps = geo_fps + geo_fps_denmark
    pred_fps = pred_fps + pred_fps_denmark
    bboxs = bboxs + bboxs_denmark

    fps = list(zip(img_fps, geo_fps, pred_fps))
    stats, _ = calculate_cmat(fps, bboxs=bboxs, df_fp='./tmp/external_evaluation_bbox.csv')
    pprint(stats)
    print('done')