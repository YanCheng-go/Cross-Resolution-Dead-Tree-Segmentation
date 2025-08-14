"""Evaluate the performance of weighted sampler and export the information of patches after generating processed_dir"""

from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
import geopandas as gpd

from src.data.base_dataset import get_dataloader
from src.data.collate import default_collate_with_shapely_support
from train.base import init_config
from train.ordinal_watershed import weightedSampler
from train.segmentation import SegmentationTrainer
import scienceplots
plt.style.use(['science', 'nature'])


def data_distribution(data, col='patch_hectares', transform_func=lambda x: np.sqrt(x / 256 / 256 * 10000),
                      out_dir=None, prefix='', suffix='', bins=30, width=0.04, xlabel='image resolution (m)',
                      ylabel='number of patches\n(256 * 256 pixels)'):
    """Check the data distribution after using dataloader"""
    transform_func(data[col]).plot.hist(bins=bins, alpha=0.5, width=width, color='steelblue')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{prefix}data_distribution_{col}{suffix}.png'), dpi=300)

    plt.show()


def visualise_dataloader(dl, out_prefix='', with_outputs=True, out_dir='./tmp', n_epochs=1, use_split_col=True):
    os.makedirs(out_dir, exist_ok=True)
    total_num_images = len(dl) if use_split_col else len(dl.dataset.indices)
    patch_res = []
    feature_counts = []
    patch_hectares = []
    patch_id = []
    count = []
    scale_factors = []
    ori_resolution = []
    spatial_clusters = []
    ori_transform = []
    area_id = []
    geometry = []
    hectares = []
    src_name = []

    for epoch in range(n_epochs):
        for i, batch in tqdm(enumerate(dl)):
            feature_counts.extend(batch['feature_counts'].tolist())
            patch_res.extend(batch['ori_resolution'].tolist())
            patch_hectares.extend(batch['patch_hectares'].tolist())
            spatial_clusters.extend(batch['spatial_clusters'])
            patch_id.extend(batch['patch_id'].tolist())
            count.extend(batch['count'].tolist())

            if 'scale_factors' in batch.keys():
                scale_factors.extend(batch['scale_factors'].tolist())
            else:
                scale_factors.extend([1]*len(batch['count'].tolist()))

            ori_resolution.extend(batch['ori_resolution'].tolist())
            hectares.extend(batch['hectares'].tolist())
            area_id.extend(batch['area_id'].tolist())
            ori_transform.extend(batch['ori_transform'])
            geometry.extend(batch['geometry'])
            src_name.extend(batch['src_name'])

    if with_outputs:
        df = pd.DataFrame({'feature_counts': feature_counts, 'patch_res': patch_res, 'patch_hectares': patch_hectares,
                           'patch_id': patch_id, 'count': count, 'scale_factors': scale_factors,
                           'ori_resolution': ori_resolution, 'spatial_clusters': spatial_clusters,
                           'ori_transform': ori_transform, 'area_id': area_id, 'geometry': geometry,
                           'hectares': hectares, 'src_name': src_name
                           })
        df['countHa'] = df['feature_counts'] / df['patch_hectares']
        csv_fp = os.path.join(out_dir, f'{out_prefix}_patch_info.csv')
        df.to_csv(csv_fp)

        data_distribution(data=df, col='patch_res', transform_func=lambda x: x,
                          bins=200, width=0.04, xlabel='image resolution (m)',
                          out_dir=out_dir, prefix=f'{out_prefix}_')

        data_distribution(data=df, col='countHa', transform_func=lambda x: np.sqrt(x), bins=31,
                          width=0.4, xlabel='sqrt(number of dead trees per ha)',
                          out_dir=out_dir, prefix=f'{out_prefix}_')

        # plot the number of patches per spatial cluster
        plot_bars = df['spatial_clusters'].value_counts().plot(kind='bar', color='steelblue', alpha=0.5)
        plot_bars.set_ylabel('number of patches\n(256 * 256 pixels)')
        plot_bars.set_xlabel('spatial clusters')
        # plot_bars.set_xticklabels(range(len(plot_bars.get_xticks())))
        plt.savefig(os.path.join(out_dir, f'{out_prefix}_spatial_clusters.png'), dpi=300)
        plt.show()

        print(f"Num. unique images seen: {len(set(patch_id))}/{total_num_images}")

    return feature_counts, patch_res, patch_id


def weightedSampler_old(dataset):
    """Previous method for weighted sampling, not used anymore."""
    train_patches = copy(dataset.patch_df)

    assert "weight_vars" in train_patches.columns
    assert train_patches['weight_vars'].isnull().sum() == 0

    # check the unique value of weight_vars and remove empty or minority..
    # e.g. assert train_patches['spatial_clusters'].isnull().sum() == 0. Otherwise, replace NaN in the dataframe
    weight_vars = np.unique(train_patches['weight_vars'])
    weight_vars = weight_vars[0] if len(weight_vars) == 1 else None
    weight_col = []

    if "spatial_clusters" in weight_vars:
        train_patches = pd.DataFrame(train_patches)
        train_patches['count'] = train_patches.groupby('spatial_clusters')['area_id'].transform('size')
        train_patches['cluster_weight'] = (
                train_patches.groupby('spatial_clusters').count()['count'].sum() /
                train_patches['count'])
        weight_col.append('cluster_weight')

    if 'patch_hectares' in weight_vars and 'feature_counts' in weight_vars:
        # Given that there is a lower number of patches with a higher number of feature counts / ha...
        # these patches should be oversampled, i.e. higher weights
        train_patches['countHa'] = train_patches['feature_counts'] / train_patches['patch_hectares']

        if "spatial_clusters" in weight_vars:
            # to avoid negative values after applying log10
            train_patches['countHa'] = train_patches['countHa'].apply(lambda x: x + 10 if x != 0 else x)
            train_patches['countHa_weight'] = np.log10(
                train_patches['countHa'])  # needs to be adepted according to the data distribution
            train_patches['countHa'] = train_patches['countHa'].apply(lambda x: x - 10 if x != 0 else x)
            # replace -inf to np.nan -> for patches with 0 features, the log10 is -np.inf
            train_patches['countHa_weight'] = train_patches['countHa_weight'].replace([np.inf, -np.inf], np.nan)
            train_patches.loc[train_patches['countHa_weight'].isnull(), 'countHa_weight'] = \
                (train_patches['countHa_weight'][~train_patches['countHa_weight'].isnull()].sum() * 0.01 /
                 train_patches['countHa_weight'].isnull().sum())
            # # # For patches with no features, use the 10th percentile of countHa_weight for each spatial clusters
            # train_patches['countHa_weight'] = (train_patches.groupby('spatial_clusters')['countHa_weight']
            #                                    .transform(lambda x: x.fillna(x.quantile(0.05))))
            # # # For clusters, where all patches do not have features -> feature count equal to 0
            # train_patches['countHa_weight'] = train_patches['countHa_weight'].fillna(train_patches[train_patches['countHa'] == 0]['countHa_weight'].median())
            weight_col.append('countHa_weight')

            # Nested weighting, i.e. depending on multiple characteristics of the patches, in this case,
            # the number of patches per spatial cluster and the number of features per hectare
            sum_countHa_weight = train_patches.groupby('spatial_clusters')['countHa_weight'].transform(
                'sum')
            train_patches['cluster_countha_weight'] = (train_patches['countHa_weight'] / sum_countHa_weight *
                                                       train_patches['cluster_weight'])
            weight_col.append('cluster_countha_weight')
        else:
            train_patches['countHa_weight'] = np.log10(train_patches['countHa'] + 10)
            weight_col.append('countHa_weight')


    elif 'patch_hectares' in weight_vars:
        train_patches['hectare_weight'] = 1 / train_patches['patch_hectares']
        weight_col.append('hectare_weight')

    elif 'feature_counts' in weight_vars:
        train_patches['count_weight'] = np.log10(train_patches['feature_counts'] + 10)
        weight_col.append('count_weight')

    used_weight_var = 'cluster_countha_weight'  # can be any one in wight_col
    assert 'cluster_countha_weight' in weight_col

    # data_distribution(data=train_patches, col='patch_hectares', transform_func=lambda x: np.sqrt(x / 256 / 256 * 10000),
    #                   bins=200, width=0.04, xlabel='image resolution (m)',
    #                   out_dir='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/forEGU2024')

    sampler = WeightedRandomSampler(train_patches[used_weight_var].tolist(), int(2 * len(dataset)),
                                    replacement=True)
    return sampler


def run(cls, weighted_sampling=False, n_epochs=1, data_date='20240430', out_dir=None):
    """Export data distributions and the information of patches (only when the weighted is False)
    for datasets in processed_dir"""

    config = init_config(cls)
    data, config = SegmentationTrainer.init_data(config)
    assert config.target_classes is None or \
           len(config.target_classes) == config.n_classes or \
           (len(config.target_classes) == 2 and config.n_classes == 1), \
        "Different class count found. \n" \
        "Possible reason:" \
        "\t Config class names ('target_classes') and " \
        "config number of classes ('n_classes') do not align "

    train_dataset, val_dataset, test_dataset = SegmentationTrainer.init_dataset_and_split(config, **data)

    user_split_col = True if config.dataset_split_by != 'patches' else False

    out_dir = f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports/forEGU2024/data{data_date}' if out_dir is None else out_dir
    os.makedirs(out_dir, exist_ok=True)

    if weighted_sampling:
        visualise_dataloader(get_dataloader(
            dataset=train_dataset, batch_size=int(len(train_dataset)*2), num_workers=32,
            collate_fn=default_collate_with_shapely_support, train=False,
            sampler=weightedSampler(train_dataset, config.dataset_split_by), shuffle=False,
        ), out_prefix='sampler_trainset', with_outputs=True, out_dir=out_dir, n_epochs=n_epochs,
            use_split_col=user_split_col)
    else:
        # dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
        visualise_dataloader(get_dataloader(
            train_dataset, 64, 32,
            collate_fn=default_collate_with_shapely_support, train=False, sampler=None, shuffle=False
        ), out_prefix='nosampler_trainset', with_outputs=True, out_dir=out_dir, use_split_col=user_split_col)
        visualise_dataloader(get_dataloader(
            val_dataset, 64, 32,
            collate_fn=default_collate_with_shapely_support, train=False, sampler=None, shuffle=False
        ), out_prefix='nosampler_valset', with_outputs=True, out_dir=out_dir, use_split_col=user_split_col)

        import ipdb; ipdb.set_trace()
        visualise_dataloader(get_dataloader(
            test_dataset, 64, 32,
            collate_fn=default_collate_with_shapely_support, train=False, sampler=None, shuffle=False,
        ), out_prefix='nosampler_testset', with_outputs=True, out_dir=out_dir, use_split_col=user_split_col)
        df1 = pd.read_csv(Path(out_dir) / 'nosampler_testset_patch_info.csv')
        df1['split_sp'] = 'test'
        df2 = pd.read_csv(Path(out_dir) / 'nosampler_valset_patch_info.csv')
        df2['split_sp'] = 'val'
        df3 = pd.read_csv(Path(out_dir) / 'nosampler_trainset_patch_info.csv')
        df3['split_sp'] = 'train'
        df_all = pd.concat([df1, df2, df3])
        df_all.to_csv(Path(out_dir) / 'nosampler_all_patch_info.csv')
        df_all = pd.read_csv(Path(out_dir) / 'nosampler_all_patch_info.csv')
        df_all = df_all.reset_index().rename(columns={'index': 'unique_id'})
        df_all["geometry"] = gpd.GeoSeries.from_wkt(df_all["geometry"])
        df_all = gpd.GeoDataFrame(df_all, crs='EPSG:4326', geometry='geometry')
        df_all.to_file(Path(out_dir) /'nosampler_all_patch_info.gpkg', driver='GPKG')


if __name__ == '__main__':
    from config.treehealth_5c import TreeHealthSegmentationConfig

    data_date = '20240717_CH'

    processed_dir = f'/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/{data_date}_256_withFin_res_ha_samplerInfo_noPart_5c'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240502_data20240502_256_DE_CH'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240508_data20240508_256_noFin_noDropNIR'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240502_data20240502_256_ES'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240515_data20240508_256_3c_removeEdge2'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240521_data20240508_256_noES_removeEdge'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240531_data20240522_256_5c_countWeights'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240502_data20240502_256_DE_CH'
    processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240719_data20240717_256_CH_countWeights_edgeWeights'

    cls = TreeHealthSegmentationConfig()
    cls.__dict__.update({
        'processed_dir': processed_dir,
        'split_col': 'dataset_sp',
        'dataset_split_by': "areas",
        "save_samples": True,
        'reference_source': 'germany20cm_2022'
    })
    out_dir = None

    #
    # # for drone images
    # from config.treehealth_spainDrone import TreeHealthSegmentationConfig
    # processed_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Spain_uav/processed_dir/aug20240524_data20240524_labeledArea6_removeEdge'
    # cls = TreeHealthSegmentationConfig()
    # cls.__dict__.update({
    #     'processed_dir': processed_dir,
    #     'split_col': None,
    #     'dataset_split_by': "patches",
    #     "save_samples": True,
    #     'reference_source': 'drone_spain_25cm_2023'
    # })
    # out_dir = '/mnt/raid5/DL_TreeHealth_Aerial/Spain_uav/reports/forEGU24/data20240524'

    # Visualize the data distributions after using weighted sampler,
    # Export the information of patches for datasets in processed_dir, which are used in evaluation plots
    run(cls=cls, weighted_sampling=False, n_epochs=1, data_date=data_date, out_dir=out_dir)

    # Visualize the data distributions after using weighted sampler
    # run(cls=cls, weighted_sampling=True, n_epochs=6, data_date=data_date, out_dir=out_dir)