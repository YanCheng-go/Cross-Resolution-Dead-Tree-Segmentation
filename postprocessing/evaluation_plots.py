"""visualize the evaluation results"""
import glob
import itertools
import os
from pathlib import Path

import pandas as pd

# residual plot
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize
from openpyxl.styles.builtins import title
from scipy import stats
import numpy as np
import geopandas as gpd

import scienceplots

plt.style.use(['science', 'nature'])

# Evaluation by labeled areas and number of pixels...

def plot_residuals(df, x, y, hue=None, save_fp=None, color='b'):
    sns.set_theme(style="whitegrid")
    sns.residplot(data=df, x=x, y=y, lowess=True, color=color, scatter_kws={'s': 10}, line_kws={'color': 'red', 'lw': 1})
    sns.lmplot(data=df, x=x, y=y, hue=hue, scatter_kws={'s': 10}, palette='tab10')
    plt.xlabel(x)
    plt.ylabel(y)
    # if hue:
    #     plt.legend(title=hue)
    if save_fp:
        plt.savefig(save_fp, dpi=300)
    # plt.show()


def plot_lmplot(df, x, y, hue=None, save_fp=None, palette=None, per_ha=True):
    with plt.style.context(['science', 'nature']):
        # set font family to sans-serif
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'
        plt.rcParams['axes.linewidth'] = 1
        # remove inf or none values from df
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])

        if per_ha:
            df['Label'] = df['Label'] / df['hectares_x']
            df['Prediction'] = df['Prediction'] / df['hectares_x']

        # set palette based on the values of the hue column, user defined for the lmplot in seaborn
        palette_ = sns.color_palette("tab10") if not palette else palette
        # remove outliers in where the difference between the values of x and y are more than 3 standard deviations
        g = sns.lmplot(data=df, x=x, y=y, hue=hue, scatter_kws={'s': 10}, palette=palette_, legend=False)
        # set color based on the group values for the scattered plot

        # remove legend
        plt.legend([], [], frameon=False)
        if per_ha:
            plt.xlabel(f'{x}' + ' (count per ha)')
            plt.ylabel(f'{y}' + ' (count per ha)')
        else:
            plt.xlabel(f'{x}' + ' (count per patch)')
            plt.ylabel(f'{y}' + ' (count per patch)')

        # add a diagonal line
        x_line = np.linspace(df[x].min(), df[x].max(), 100)
        plt.plot(x_line, x_line, color='red', linestyle='--')

        # add r2 to the plot and remove inf or none values and place it at the top left corner without overlapping with
        # the data points or the legend

        # Debug...
        try:
            r2 = stats.pearsonr(df[x], df[y])[0] ** 2
        except:
            return
        plt.text(df[x].min(), df[y].max(), '$R^2$={:.2f}'.format(r2), color='black', ha='left', va='top', fontdict={'size': 10})

        if hue:
            plt.legend(title=None, loc='lower right')
            if palette is not None:
                map_dict = dict(zip(range(5), ['Alps Conifer and Mixed Forests',
                                               'Mediterranean Forests, Woodlands and Scrub',
                                               'Temperate Conifer Forests',
                                               'Temperate Broadleaf and Mixed Forests',
                                               'Boreal Forests or Taiga']))
                handles, labels = g.ax.get_legend_handles_labels()
                g.ax.legend(loc='lower right', handles=handles[:],
                            labels=[map_dict.get(int(i)) for i in labels], title=None, facecolor='white',
                            frameon=True)

            # calculate r2 and rMAE and bias(%) for each hue group
            if hue == 'Patch resolution':
                stats_str = []
                for i, group in enumerate([0.12, 0.2, 0.3, 0.4, 0.5, 0.6]):
                    df_group = df[df[hue] == group]
                    if df_group.empty:
                        continue
                    r2 = stats.pearsonr(df_group[x], df_group[y])[0] ** 2
                    mae = np.abs(df_group[x] - df_group[y]).mean()
                    bias = sum((df_group[x] - df_group[y])) / sum(df_group[x]) * 100
                    stats_str.append(f': {r2:.2f}, {mae:.2f}, {bias:.1f}')
                # change the legend labels
                map_dict = dict(zip([0.12, 0.2, 0.3, 0.4, 0.5, 0.6], ['0.12', '0.2-0.25', '0.3', '0.4', '0.5', '0.6']))
                handles, labels = g.ax.get_legend_handles_labels()
                g.ax.legend(loc='upper left', handles=handles[:], bbox_to_anchor=(0, 0.9),
                            labels=[''.join([map_dict.get(float(i)), j]) for i, j in zip(labels, stats_str)], title=None, facecolor='white',
                            frameon=False)

                # update title of the legend
                g.ax.get_legend().set_title('Resolution: $R^2$, MAE, bias', prop={'size': 10})
                # update the fontsize for legend
                for item in g.ax.get_legend().get_texts():
                    item.set_fontsize(10)
                # update the fontsize for all elements in the plot to 10
                for item in ([g.ax.title, g.ax.xaxis.label, g.ax.yaxis.label] +
                                g.ax.get_xticklabels() + g.ax.get_yticklabels()):
                        item.set_fontsize(10)

        if save_fp:
            # set figsize
            plt.gcf().set_size_inches(4.5, 4.5)
            plt.savefig(save_fp, dpi=300)
        # plt.show()


def heatmap(df, x, y, save_fp=None):
    sns.set_theme(style="whitegrid")
    df = df.pivot_table(index=x, columns=y, values='r2')
    sns.heatmap(df, annot=True, fmt=".2f")
    if save_fp:
        plt.savefig(save_fp, dpi=300)
    # plt.show()


def plot_with_residuals_and_histograms(df, x, y, hue=None):
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create grid: 3 rows, 3 columns
    # Top right and bottom left will be histograms of x and y, respectively.
    # Bottom right will be the scatter plot with residuals.

    # Histogram for x-variable
    plt.subplot(3, 3, 1)  # first row, first column
    sns.histplot(df[x], kde=False, color='skyblue')
    plt.title(f'Histogram of {x}')

    # Scatter plot with regression line
    plt.subplot(3, 3, 4)  # second row, first column, spans all columns
    scatter_ax = sns.scatterplot(data=df, x=x, y=y, hue=hue)
    sns.regplot(data=df, x=x, y=y, scatter=False, ax=scatter_ax)
    plt.title(f'Scatter plot of {x} vs. {y}')

    # Residual plot
    plt.subplot(3, 3, 7)  # third row, first column, spans all columns
    sns.residplot(data=df, x=x, y=y, scatter_kws={'s': 10})
    plt.title(f'Residuals of {y} against {x}')

    # Histogram for y-variable
    plt.subplot(3, 3, 6)  # second row, third column
    sns.histplot(df[y], kde=False, orientation='horizontal', color='salmon')
    plt.title(f'Histogram of {y}')

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def prepare_data(in_folder_name, data_date='20240430', save_df=False, dataset_name=['val', 'test', 'train'],
                 split_resample=False, rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports', with_biomes=True,
                 no_downsampled=True):

    wd = '{}/{}'.format(rd, in_folder_name)

    patch_info_fp = glob.glob(f'{rd}/forEGU2024/data{data_date}/nosampler_all_patch_info_ecoregions.*')[0] \
        if with_biomes else glob.glob(f'{rd}/forEGU2024/data{data_date}/nosampler_all_patch_info.gpkg')[0]

    if patch_info_fp.endswith('gpkg'):
        try:
            patch_info = gpd.read_file(patch_info_fp, driver='GPKG', layer=1)
            if len(patch_info) == 0:
                patch_info = gpd.read_file(patch_info_fp, driver='GPKG', layer=0)
        except:
            patch_info = gpd.read_file(patch_info_fp, driver='GPKG')
    else:
        patch_info = gpd.read_file(patch_info_fp)

    patch_info['biome_regroup'] = '0'

    if with_biomes:
        # Ecoregions
        patch_info.drop_duplicates(subset=['unique_id'], keep="first", inplace=True)
        patch_info['biome_regroup'] = patch_info.apply(lambda x: x['ECO_NAME'] if x['ECO_NAME'] in ['Alps conifer and mixed forests'] else x['BIOME_NAME'], axis=1)
        replace_dict = dict(zip(
            ['Alps conifer and mixed forests', 'Mediterranean Forests, Woodlands & Scrub', 'Temperate Conifer Forests',
             'Temperate Broadleaf & Mixed Forests', 'Boreal Forests/Taiga'],
            ['Alps conifer and mixed forests', 'Mediterranean Forests, Woodlands and Scrub', 'Temperate Conifer Forests',
             'Temperate Broadleaf and Mixed Forests', 'Boreal Forests or Taiga']
        ))
        map_dict = dict(zip(replace_dict.values(), range(len(replace_dict.values()))))
        patch_info['biome_regroup'] = patch_info['biome_regroup'].replace(replace_dict)
        patch_info['biome_regroup_rename'] = patch_info['biome_regroup']
        patch_info['biome_regroup'] = patch_info['biome_regroup'].map(map_dict)
        patch_info['biome_regroup'] = patch_info['biome_regroup'].astype(str)

    #fp_list = [i for i in os.listdir(wd) if i.endswith('.csv') if 'predictions' not in i]
    fp_list = [i for i in os.listdir(wd) for dataset_n in dataset_name
               if i.endswith(f'post-watershed_count-bias_label-vs-prediction_{dataset_n}.gpkg')][:len(dataset_name)]
    fp_list_shp = False if fp_list[0].endswith('csv') else True

    df_list = []
    for fp in fp_list:
        df_list.append(gpd.read_file(os.path.join(wd, fp)))
    df = pd.concat(df_list)

    if fp_list_shp:
        df.drop(columns=['patch_id', 'area_id'], inplace=True)

    if '_removeEdge' in in_folder_name and 'Label2' in df.columns and df['Label2'].sum() > 0:
        df[['Label', 'Prediction', 'pred_area', 'label_area', 'hectares', 'iou_1', 'iou_0', 'Patch resolution', 'Difference', 'count_loss_ha']] = (
            df[['Label2', 'Prediction', 'pred_area', 'label_area', 'hectares', 'iou_1', 'iou_0',
                'Patch resolution', 'Difference2', 'count_loss_ha2']].astype(float))
    else:
        df[['Label', 'Prediction', 'pred_area', 'label_area', 'hectares', 'iou_1', 'iou_0', 'Patch resolution']] = (
            df[['Label', 'Prediction', 'pred_area', 'label_area', 'hectares', 'iou_1', 'iou_0',
                'Patch resolution']].astype(float))

    df['Patch ID'] = df['Patch ID'].astype(float).astype(int)
    patch_info['patch_id'] = patch_info['patch_id'].astype(int)
    df = df.merge(patch_info[[i for i in patch_info.columns if 'geometry' not in i and 'unnamed' not in i]],
                  left_on=['Patch ID', 'Class'], right_on=['patch_id', 'split_sp'], how='left')


    # replace values in the spatial cluster column
    lookup_spatial_clusters = {'california60cm_2020': 'CA', 'finland40cm_2021': 'FI', 'germany20cm_2022': 'DE',
                               'swiss25cm_2022': 'CH', 'spain25cm_2022': 'ES', 'swiss20cm_2022': 'CH',
                               'drone_spain_25cm_2023': 'ES_drone'}

    df['spatial_clusters'] = df['src_name_x'].map(lookup_spatial_clusters)

    # remove inf or none values from df
    df = df[~((df['label_area'] == 0) & (df['Label'] != 0))]
    df = df.dropna(subset=['Label', 'Prediction', 'label_area', 'hectares_x'])
    df = df[df['hectares_x'] >= 1 / 3 * df['ori_resolution'] ** 2 * 256 ** 2 / 10000]

    df['pred_area'] = df['pred_area'] * df['Patch resolution'] ** 2 / 10000
    df['label_area'] = df['label_area'] * df['Patch resolution'] ** 2 / 10000

    # Calculate count per hectare loss
    df['Label'] = df['Label']
    df['Prediction'] = df['Prediction']
    df['Residual'] = df['Label'] - df['Prediction']
    df['count_loss_patch'] = df['Prediction'] - df['Label']
    df['mIoU'] = (df['iou_1'] + df['iou_0']) / 2
    df['area_loss'] = (df['pred_area'] - df['label_area']) / df['label_area'] * 100

    # replace values in the 'Patch resolution' column
    df['Patch resolution'] = df.apply(lambda x: np.round(x['Patch resolution'], 2), axis=1)
    df['Patch resolution'] = df['Patch resolution'].replace({0.61: 0.60, 0.59: 0.60, 0.58: 0.60})
    df['resampled'] = df.apply(lambda x: 1 if abs(np.round(x['ori_resolution'], 2) - np.round(x['Patch resolution'], 2)) > 0.02 else 0, axis=1)
    df['downsampled'] = df.apply(lambda x: 1 if np.round(x['ori_resolution'],2) <= np.round(x['Patch resolution'],2) else 0, axis=1)

    if no_downsampled:
        df = df[df['downsampled'] == 1]

    if save_df and fp_list_shp:
        out_fp = Path(wd) / 'data_enriched_{}.gpkg'.format('_'.join(dataset_name) if len(dataset_name) > 1 else dataset_name[0])
        df.to_file(out_fp, driver='GPKG')
        if split_resample:
            out_fp = Path(wd) / 'data_enriched_{}_nonresample.gpkg'.format(
                '_'.join(dataset_name) if len(dataset_name) > 1 else dataset_name[0])
            df[df['resampled'] == 0].to_file(out_fp, driver='GPKG')

    df = df[~((df['patch_id'] == 156) & (df['Class'] == 'test'))]

    return df, dataset_name


def calculalte_r2_per_group(df, plot=False, out_folder_name=None, group_col='spatial_clusters',
                            dataset_name=['val', 'test'], per_ha=True, rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports'):
    # data = {
    # "spatial_clusters": ["DE", "CA", "CH", "ES", "FI", "DE", "CH", "ES", "FI", "DE", "CH", "ES", "DE", "CH", "DE", "CH", "ES", "FI", "CH", "ES", "CH"],
    # "Patch_resolution": [0.60, 0.60, 0.60, 0.60, 0.60, 0.40, 0.40, 0.40, 0.40, 0.30, 0.30, 0.30, 0.20, 0.20, 0.50, 0.50, 0.50, 0.50, 0.25, 0.25, 0.10],
    # "r2": [0.924091, 0.982547, 0.817858, 0.319743, 0.863970, 0.971049, 0.849888, 0.488616, 0.877532, 0.979557, 0.870465, 0.566373, 0.988876, 0.967074, 0.959558, 0.826534, 0.391834, 0.907208, 0.582261, 0.554982, 0.882159]
    # }

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Label', 'Prediction'])

    if per_ha:
        df['Label'] = df['Label'] / df['hectares_x']
        df['Prediction'] = df['Prediction'] / df['hectares_x']

    df['ori_res'] = df['ori_transform'].apply(lambda x: x.split('| ')[1].split(',')[0])

    # for each resolution and spatial cluster
    r2_dict = []
    for res, reg, ori in itertools.product(df['Patch resolution'].unique(), df[group_col].unique(), df['ori_res'].unique()):
        df_in = df[(df['Class'].isin(dataset_name)) & (df['Patch resolution'] == res) & (df[group_col] == reg) & (df['ori_res'] == ori)]
        df_in.dropna(subset=['Label', 'Prediction'], inplace=True)
        if df_in.empty:
            continue
        r2 = stats.pearsonr(df_in['Label'], df_in['Prediction'])[0] ** 2
        r2_dict.append([reg, res, ori, r2])

        if plot:
            plot_lmplot(df_in,'Label', 'Prediction', hue=None,
                        save_fp=f'{rd}/forEGU2024/{out_folder_name}/residua'
                                'l_plot{}_{}_{}_{}.png'.format('_per_patch' if not per_ha else '', '_'.join(dataset_name) if len(dataset_name) > 1 else dataset_name[0], res, reg), per_ha=per_ha)

    r2_df = pd.DataFrame(r2_dict, columns=['Spatial group', 'Patch resolution', 'ori_res', 'r2'])

    # combine column 'Spatial group' and 'Scale' to 'Spatial group' only when
    r2_df['Spatial group'] = r2_df['Spatial group'] + '_' + r2_df['ori_res']
    r2_df = r2_df.drop(columns=['ori_res'])

    return r2_df


def calculalte_metric_per_group(df, plot=False, out_folder_name=None, col='mIoU', group_col='spatial_clusters',
                                dataset_name=['val', 'test'], rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports'):

    if col == 'mIoU':
        df = df.dropna(subset=['iou_1', 'iou_0'])
        df = df[(df['label_area'] != 0) | (df['Label'] != 0)]
    if col == 'iou_1':
        df = df.dropna(subset=['iou_1'])
        df = df[(df['label_area'] != 0) | (df['Label'] != 0)]
    if col == 'background_overestimate':
        df = df[(df['Label'] == 0)]
        df['background_overestimate'] = (df['pred_area'] / df['hectares_x'] * 100)
    if col == 'area_loss':
        df = df.dropna(subset=['label_area'])
        df = df[(df['label_area'] != 0) | (df['Label'] != 0)]
        # convert the area loss to percentage with 0 decimal points
        df['area_loss'] = df['area_loss'].round(0)
    if col == 'count_loss_ha' or col == 'count_loss_patch':
        df['count_loss_ha'] = pd.to_numeric(df['count_loss_ha'])
        df['count_loss_patch'] = pd.to_numeric(df['count_loss_patch'])
        df = df.dropna(subset=['count_loss_ha', 'count_loss_patch'])
        df = df[(df['Label'] != 0) | (df['Label'] != 0)]
    if col == 'count_bias':
        df_group = df.groupby(['Class', 'Patch resolution', group_col]).agg({'Label': 'sum', 'Prediction': 'sum'}).reset_index()
        df_group['count_bias'] = (df_group['Label'] - df_group['Prediction']) / df_group['Label'] * 100
        df_group = df_group.rename(columns={group_col: 'Spatial group'})
        return df_group[['Spatial group', 'Patch resolution', col]]

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Label', 'Prediction'])

    df['ori_res'] = df['ori_transform'].apply(lambda x: x.split('| ')[1].split(',')[0])

    # for each resolution and spatial cluster
    miou_dict = []
    test_ = []
    for res, reg, ori in itertools.product(df['Patch resolution'].unique(), df[group_col].unique(), df['ori_res'].unique()):
        df_in = df[(df['Class'].isin(dataset_name)) & (df['Patch resolution'] == res) & (df[group_col] == reg) & (df['ori_res'] == ori)]
        df_in.dropna(subset=['Label', 'Prediction'], inplace=True)
        if df_in.empty:
            continue
        miou_dict.append([reg, res, ori, df_in[col].mean()])
        test_.append([reg, res, ori, len(df_in['Patch resolution'])])
        if plot:
            plot_lmplot(df_in, 'Label', 'Prediction', hue=None,
                        save_fp=f'{rd}/forEGU2024/{out_folder_name}/residua'
                                'l_plot_{}_{}_{}.png'.format('_'.join(dataset_name) if len(dataset_name) > 1 else dataset_name[0], res, reg))
    out_df = pd.DataFrame(miou_dict, columns=['Spatial group', 'Patch resolution', 'ori_res', col])
    # combine column 'Spatial group' and 'Scale' to 'Spatial group' only when
    out_df['Spatial group'] = out_df['Spatial group'] + '_' + out_df['ori_res']
    out_df = out_df.drop(columns=['ori_res'])
    print(out_df)
    print(pd.DataFrame(test_))
    return out_df


def plot_lmplot_all(out_folder_name, in_folder_name, spatial_cluster=False, resolution=False,
                    spatial_group='spatial_clusters', dataset_name=['val', 'test'], data_date='20240430', per_ha=True,
                    rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports', with_biomes=True, no_downsampled=True):

    os.makedirs('{}/forEGU2024/{}'.format(rd, out_folder_name), exist_ok=True)

    df, dataset_name = prepare_data(in_folder_name, data_date=data_date, dataset_name=dataset_name, rd=rd,
                                    with_biomes=with_biomes, no_downsampled=no_downsampled)

    # visualize the residuals
    # for all data
    if dataset_name == ['val', 'test', 'train']:
        plot_lmplot(df, 'Label', 'Prediction', hue='Class', palette=None,
                    save_fp='{}/forEGU2024/{}/residual_plot{}_all.png'.format(rd, out_folder_name, '_per_patch' if per_ha is False else ''), per_ha=per_ha)

    else:
        plot_lmplot(df[(df['Class'].isin(dataset_name)) & (df['resampled'] != 1)], 'Label', 'Prediction', hue=spatial_group,
                        save_fp=f'{rd}/forEGU2024/{out_folder_name}/residual_p'
                                'lot{}_{}_{}.png'.format('_per_patch' if per_ha is False else '', '_'.join(dataset_name) if len(dataset_name) > 1 else dataset_name[0], spatial_group),
                    palette=None if spatial_group == 'spatial_clusters' else {'0': '#1f77b4', '1': '#d62728', '2': '#2ca02c', '3': '#ff7f0e', '4': '#9467bd'}, per_ha=per_ha)
    # for each spatial cluster
    if spatial_cluster:
        for sg in df[spatial_group].unique():
            plot_lmplot(df[(df['Class'].isin(dataset_name)) & (df[spatial_group] == sg) & (df['resampled'] != 1)],
                        'Label', 'Prediction', hue='Patch resolution',
                        save_fp=f'{rd}/forEGU2024/{out_folder_name}/residual_p'
                                'lot{}_{}_{}.png'.format('_per_patch' if per_ha is False else '', '_'.join(dataset_name) if len(dataset_name) > 1 else dataset_name[0], sg), palette=None, per_ha=per_ha)

    # for each resolution and spatial cluster
    if resolution and spatial_cluster:
        calculalte_r2_per_group(df, plot=True, out_folder_name=out_folder_name, group_col=spatial_group, per_ha=per_ha, rd=rd)


def plot_by_resolutions(out_folder_name, in_folder_name, ori_res_list=['0.12', '0.20', '0.25'],
                        spatial_group_list=['DE', 'CH', 'ES'], spatial_group='spatial_clusters',
                        dataset_name=['val', 'test'], data_date='20240430', per_ha=True,
                        rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports', with_biomes=True, no_downsampled=True,
                        fig_name='lmplot_by_resolution', patch_res_map=None):
    """ Plot the lmplot for each resolution """

    os.makedirs('{}/forEGU2024/{}'.format(rd, out_folder_name), exist_ok=True)

    df, dataset_name = prepare_data(in_folder_name, data_date=data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)

    df['ori_res'] = df['ori_transform'].apply(lambda x: x.split('| ')[1].split(',')[0])

    if patch_res_map:
        df['Patch resolution'] = df['Patch resolution'].map(lambda x: patch_res_map.get(x, x))

    in_df = df[(df['Class'].isin(dataset_name)) & (df['ori_res'].isin(ori_res_list)) & (df[spatial_group].isin(spatial_group_list))]
    plot_lmplot(in_df, 'Label', 'Prediction', hue='Patch resolution', save_fp=f'{rd}/forEGU2024/{out_folder_name}/{fig_name}.png',
                palette=None if spatial_group == 'spatial_clusters' else {'0': '#1f77b4', '1': '#d62728', '2': '#2ca02c', '3': '#ff7f0e', '4': '#9467bd'}, per_ha=per_ha)

    return


# warp the function named as plot_heatmap
def plot_heatmap(out_folder_name, df, var_name='mIoU', dataset_name=['val', 'test'],
                 rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports', patch_res_map=None):
    os.makedirs('{}/forEGU2024/{}'.format(rd, out_folder_name), exist_ok=True)

    # Data preparation
    val_col = [i for i in df.columns if i not in ['Spatial group', 'Patch resolution']][0]

    # Filtering out the specified resolution
    # df_filtered = df[df['Patch resolution'] != 0.25]

    if patch_res_map:
        df['Patch resolution'] = df['Patch resolution'].map(lambda x: patch_res_map.get(x, x))

    # Pivoting the DataFrame for the heatmap
    heatmap_data_filtered = df.pivot_table(index='Spatial group', columns='Patch resolution',
                                           values=val_col, aggfunc='mean')

    # Creating the heatmap with the seaborn-whitegrid style
    fig, ax = plt.subplots()
    if var_name == 'count_bias':
        fmt = ".1f"
    else:
        fmt = ".2f"
    heatmap = sns.heatmap(heatmap_data_filtered, annot=True, cmap="viridis", fmt=fmt, ax=ax)
    # change the font size of the heatmap
    for text in heatmap.texts:
        text.set_fontsize(10)

    plt.title('Percentage of {}'.format(var_name)) if var_name == 'area loss' or var_name == 'background overestimate' \
        else plt.title(f'{var_name}')
    # plt.ylabel('Biome')
    plt.yticks(rotation=90)
    plt.xlabel('Image Resolution (m)')
    ax.tick_params(axis='both', which='both', length=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(which='both', length=0)

    # change y-axis labels
    label_map = {'DE_0.20': 'DE', 'CA_0.60': 'CA', 'CH_0.12': 'CH12', 'ES_0.25': 'ES', 'FI_0.40': 'FI', 'CH_0.25': 'CH25'}
    ax.set_yticklabels([label_map.get(i) for i in heatmap_data_filtered.index])

    # update x-axis labels, change 0.2 to 0.2-0.25
    ax.set_xticklabels([f'{i:.2f}' if i != 0.2 else '0.2-0.25' for i in heatmap_data_filtered.columns])

    # plt.show()
    # set figsize
    plt.gcf().set_size_inches(5, 4.5)
    # set font size for all elements in the plot to 10
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)


    plt.savefig(f'{rd}/forEGU2024/{out_folder_name}/{val_col}_'
                'reslution_biome_heatmap_{}.png'.format('_'.join(dataset_name) if len(dataset_name)>1 else dataset_name[0]), dpi=300)


def residual_histogram(in_folder_name, out_folder_name, x='Label', y='Residual', hue='Patch resolution',
                       dataset_name=['val', 'test'], data_date='20240430', resampled=True, downsampled=False,
                       plus_ori=False,
                       separate_spatial_clusters='', per_ha=True, rd='/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports',
                       with_biomes=True):
    os.makedirs('{}/forEGU2024/{}'.format(rd, out_folder_name), exist_ok=True)

    df, dataset_name = prepare_data(in_folder_name, data_date=data_date, dataset_name=dataset_name, rd=rd,
                                    with_biomes=with_biomes)

    if hue == 'combined':
        df['hue'] = (df['Patch resolution'].astype(str) + df['spatial_clusters'])
    if hue == 'Patch resolution':
        df['hue'] = df['Patch resolution']
    if hue == 'spatial_clusters':
        df['hue'] = df['spatial_clusters']
    if hue == 'Class':
        df['hue'] = df['Class']
    if hue == 'biome_regroup':
        df['hue'] = df['biome_regroup']
    df['hue'] = df['hue'].astype('str')

    df_copy = df.copy()

    if resampled:
        df = df[(df['resampled'] == 1)]
        if downsampled:
            df = df[(df['downsampled'] == 1)]
        if plus_ori:
            df = pd.concat([df, df_copy[(df_copy['resampled'] == 0)]])
    else:
        df = df[(df['resampled'] == 0)]

    if separate_spatial_clusters is not None and separate_spatial_clusters != '':
        df = df[df['spatial_clusters'] == separate_spatial_clusters]
        if len(df) == 0:
            return

    # df = df[df['Label'] > 0]

    if per_ha:
        df['Label'] = df['Label'] / df['hectares_x']
        df['Residual'] = df['Residual'] / df['hectares_x']
        df['Prediction'] = df['Prediction'] / df['hectares_x']

    # cut off the outliers based on the Label and Residual columns percentiles
    df = df[(df['Label'] < df['Label'].quantile(0.99)) & (df['Residual'] > df['Residual'].quantile(0.01)) &
            (df['Residual'] < df['Residual'].quantile(0.99))]

    sns.set_theme(style="ticks")
    palette = {'0': '#1f77b4', '1': '#d62728', '2': '#2ca02c', '3': '#ff7f0e', '4': '#9467bd'}
    # palette = {'ES': '#1f77b4', 'DE': '#d62728', 'CA': '#2ca02c', 'CH': '#ff7f0e', 'FI': '#9467bd'}

    g = sns.JointGrid(data=df, x=x, y=y, hue='hue', palette=palette)
    g.ax_joint.axvspan(df[x].mean() - df[x].std(), df[x].mean() + df[x].std(), color='skyblue', alpha=0.3, edgecolor='white')
    g.ax_joint.axhspan(df[y].mean() - df[y].std(), df[y].mean() + df[y].std(), color='skyblue', alpha=0.3, edgecolor='white')

    # # Using scatterplot for the joint and kdeplot for the margins
    g.plot(sns.scatterplot, sns.kdeplot)
    g.ax_marg_x.set_facecolor('none')

    if with_biomes:
        g.ax_joint.legend(loc='upper right', title="Biome")
        map_dict = dict(zip(range(5), ['Alps Conifer, Mixed Forests', 'Mediterranean Forests',
                                       'Temperate Conifer Forests', 'Temperate Broadleaf, Mixed Forests',
                                       'Boreal Forests or Taiga']))
        handles, labels = g.ax_joint.get_legend_handles_labels()
        g.ax_joint.legend(loc='upper right', handles=handles[:],
                          labels=[map_dict.get(int(i)) for i in labels], title=None, facecolor='white',
                          frameon=True, fontsize=11)

    if per_ha:
        g.ax_joint.set_xlabel('Label (Count per hectare)')
        g.ax_joint.set_ylabel('Residual (Count per hectare)')
    else:
        g.ax_joint.set_xlabel('Label (Count per patch)')
        g.ax_joint.set_ylabel('Residual (Count per patch)')

    # set x limit
    g.ax_joint.set_xlim(-10, 170)
    g.ax_joint.set_ylim(-25, 25)

    mean_count = df['Label'].mean()
    mean_count1 = df[df['Label'] != 0]['Label'].mean()
    std_count = df['Label'].std()
    std_count1 = df[df['Label'] != 0]['Label'].std()

    print('bias:', df['Residual'].sum() / df['Label'].sum() * 100)
    # df.groupby('hue')['Residual'].sum() / df.groupby('hue')['Label'].sum() * 100

    # rmse0 = np.sqrt(np.mean(df[df['Label'] == 0]['Residual'] ** 2))
    # mse0 = np.mean(df[df['Label'] == 0]['Residual'] ** 2)
    # mae0 = np.mean(np.abs(df[df['Label'] == 0]['Residual']))
    # g.ax_joint.text(0.95, 0.05, 'RMSE_0={:.2f}'.format(rmse0), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.10, 'MSE_0={:.2f}'.format(mse0), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.15, 'MAE_0={:.2f}'.format(mae0), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.20, 'Empty n={}'.format(len(df[df['Label'] == 0])), transform=g.ax_joint.transAxes, color='black',
    #                 ha='right')
    #
    # rmse1 = np.sqrt(np.mean(df[df['Label'] != 0]['Residual'] ** 2))
    # mse1 = np.mean(df[df['Label'] != 0]['Residual'] ** 2)
    # mae1 = np.mean(np.abs(df[df['Label'] != 0]['Residual']))
    # g.ax_joint.text(0.95, 0.30, 'RMSE_1={:.2f}'.format(rmse1), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.35, 'MSE_1={:.2f}'.format(mse1), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.40, 'MAE_1={:.2f}'.format(mae1), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.45, 'Mean(Std.)={:.2f}({:.2f})'.format(mean_count1, std_count1), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.50, 'Non-empty n={}'.format(len(df[df['Label'] != 0])), transform=g.ax_joint.transAxes, color='black',
    #                 ha='right')
    #
    # rmse = np.sqrt(np.mean(df['Residual'] ** 2))
    # mse = np.mean(df['Residual'] ** 2)
    # mae = np.mean(np.abs(df['Residual']))
    # g.ax_joint.text(0.95, 0.60, 'RMSE={:.2f}'.format(rmse), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.65, 'MSE={:.2f}'.format(mse), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.70, 'MAE={:.2f}'.format(mae), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.75, 'Mean(Std.)={:.2f}({:.2f})'.format(mean_count, std_count), transform=g.ax_joint.transAxes, color='black', ha='right')
    # g.ax_joint.text(0.95, 0.80, 'All n={}'.format(len(df)), transform=g.ax_joint.transAxes, color='black',
    #                 ha='right')

    # plt.title(f'Count loss per hectare vs. Residual')
    plt.savefig('{}/forEGU2024/{}/residual{}_histogram_{}_{}{}{}{}{}.png'
                .format(rd, out_folder_name,
                        '_per_patch' if not per_ha else '',
                        hue.replace(' ', '_'),
                        '_'.join(dataset_name),
                        '_original' if plus_ori else '',
                        '_resampled' if resampled else '',
                        '_downsampled' if downsampled else '',
                        '_'+separate_spatial_clusters if separate_spatial_clusters != '' and separate_spatial_clusters is not None else ''), dpi=300)

    # Display the plot
    # plt.show()


if __name__ == '__main__':
    # use powa envrionment otherwise latex issue
    enrich_evaluation_data = False
    _plot_lmplot = False
    plot_lmplot_by_res = False
    plot_residual_histogram = True
    plot_miou_heatmap = False
    plot_r2_heatmap = False
    plot_iou_heatmap = False
    plot_arealoss_heatmap = False
    plot_backover_heatmap = False
    plot_countlossha_heatmap = False
    plot_countlosspatch_heatmap = False
    plot_countbias_heatmap = False

    data_folder_name = 'v20241022_data20240717_autoResample'
    rd = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports'

    suffix4 = ''
    auto_resample_ = True
    no_downsampled = True
    data_date = '20240522_5c' # need to check the biome regrouping...
    group_col = 'spatial_clusters' #'spatial_clusters' or 'biome_regroup
    dataset_name = ['test', 'val']  # ['train']
    spatial_cluster_list = ['DE', 'CA', 'CH', 'ES', 'FI']

    auto_resample = True if 'autoResample' in data_folder_name or auto_resample_ else False
    suffix2 = '_ecoregions' if group_col == 'biome_regroup' else ''
    suffix1 = '_autoResample' if auto_resample else ''
    suffix3 = '_removeEdge' if 'removeEdge' in data_folder_name else ''
    figure_folder_name = '{}_data{}{}{}{}{}'.format(data_folder_name.split('_')[0], data_date, suffix1, suffix2, suffix3, suffix4)
    with_biomes = True if group_col == 'biome_regroup' else False
    patch_res_map = {0.25: 0.20}

    if plot_lmplot_by_res:
        plot_by_resolutions(out_folder_name=figure_folder_name, in_folder_name=data_folder_name,
                            ori_res_list=['0.12', '0.25', '0.20'], per_ha=False, fig_name='lmplot_by_resolution_all_2520_{}'.format('_'.join(dataset_name)),
                            patch_res_map=patch_res_map, dataset_name=dataset_name)

    if enrich_evaluation_data:
        prepare_data(in_folder_name=data_folder_name, data_date=data_date, save_df=True,
                     dataset_name=['train', 'val', 'test'], split_resample=True, rd=rd, with_biomes=with_biomes)

    if _plot_lmplot:
        per_ha = True
        if group_col == 'biome_regroup':
            plot_lmplot_all(out_folder_name=figure_folder_name, in_folder_name=data_folder_name, spatial_cluster=True,
                            resolution=True, spatial_group='biome_regroup', dataset_name=dataset_name, data_date=data_date,
                            per_ha=per_ha, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        plot_lmplot_all(out_folder_name=figure_folder_name, in_folder_name=data_folder_name, spatial_cluster=True,
                        resolution=True, spatial_group='spatial_clusters', dataset_name=dataset_name,
                        data_date=data_date, per_ha=per_ha, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        per_ha = False
        if group_col == 'biome_regroup':
            plot_lmplot_all(out_folder_name=figure_folder_name, in_folder_name=data_folder_name, spatial_cluster=True,
                            resolution=True, spatial_group='biome_regroup', dataset_name=dataset_name, data_date=data_date,
                            per_ha=per_ha, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        plot_lmplot_all(out_folder_name=figure_folder_name, in_folder_name=data_folder_name, spatial_cluster=True,
                        resolution=True, spatial_group='spatial_clusters', dataset_name=dataset_name,
                        data_date=data_date, per_ha=per_ha, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)

    if plot_residual_histogram:
        per_ha = True
        residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='spatial_clusters',
                           dataset_name=dataset_name, data_date=data_date, resampled=False, downsampled=False, plus_ori=False,
                           separate_spatial_clusters=None, per_ha=per_ha, rd=rd, with_biomes=with_biomes)
        if auto_resample:
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='spatial_clusters',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=False, plus_ori=False, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='spatial_clusters',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=True, plus_ori=False, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='spatial_clusters',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=False, plus_ori=True, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='spatial_clusters',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=True, plus_ori=True, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
        if spatial_cluster_list:
            for sc in spatial_cluster_list:
                residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name,
                                   hue='biome_regroup', dataset_name=dataset_name, data_date=data_date, resampled=False, downsampled=False, plus_ori=False,
                                   separate_spatial_clusters=sc, per_ha=per_ha, rd=rd, with_biomes=with_biomes)

        per_ha = False
        residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='biome_regroup',
                           dataset_name=dataset_name, data_date=data_date, resampled=False, downsampled=False, plus_ori=False,
                           separate_spatial_clusters=None, per_ha=per_ha, rd=rd, with_biomes=with_biomes)
        if auto_resample:
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='biome_regroup',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=False, plus_ori=False, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='biome_regroup',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=True, plus_ori=False, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='biome_regroup',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=False, plus_ori=True, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
            residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name, hue='biome_regroup',
                               dataset_name=dataset_name, data_date=data_date, resampled=True, downsampled=True, plus_ori=True, per_ha=per_ha, rd=rd,
                               with_biomes=with_biomes)
        if spatial_cluster_list:
            for sc in spatial_cluster_list:
                residual_histogram(in_folder_name=data_folder_name, out_folder_name=figure_folder_name,
                                   hue='biome_regroup', dataset_name=dataset_name, data_date=data_date, resampled=False, downsampled=False, plus_ori=False,
                                   separate_spatial_clusters=sc, per_ha=per_ha, rd=rd, with_biomes=with_biomes)

    if plot_miou_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='mIoU', group_col=group_col, dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='mIoU', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_iou_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='iou_1', group_col=group_col, dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='IoU', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_arealoss_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='area_loss', group_col=group_col, dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='area loss', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_countlossha_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='count_loss_ha', group_col=group_col, dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='count loss per hectare', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_countbias_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='count_bias', group_col=group_col, dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='count bias (%)', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_countlosspatch_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='count_loss_patch', group_col=group_col, dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='count loss per patch (256 x 256)', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_backover_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_metric_per_group(df, col='background_overestimate', group_col=group_col,
                                         dataset_name=dataset_name, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='background overestimate', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    if plot_r2_heatmap:
        df, dataset_name = prepare_data(data_folder_name, data_date, dataset_name=dataset_name, rd=rd, with_biomes=with_biomes, no_downsampled=no_downsampled)
        df = calculalte_r2_per_group(df, group_col=group_col, dataset_name=dataset_name, per_ha=False, rd=rd)
        plot_heatmap(figure_folder_name, df, var_name='$R^2$ (count per patch)', dataset_name=dataset_name, rd=rd, patch_res_map=patch_res_map)

    pass



