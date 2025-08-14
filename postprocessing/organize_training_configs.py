"""Scripts to organize training configurations and logs (ap metrics) for different datasets and models."""

import os
import json

import numpy as np
import pandas as pd
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import geopandas as gpd

plt.style.use(['science', 'nature'])

run_id_map = {'evaluation_swiss-experiments_0_20240719': 0,
              'evaluation_swiss-experiments_baseline_20240719': 0,
              'evaluation_swiss-experiments_1_20240720': 1,
              'evaluation_swiss-experiments_5_20240725': 5,
              'evaluation_swiss-experiments_4_20240724': 4,
              'evaluation_swiss-experiments_14_20240731': 14,
              'evaluation_swiss-experiments_2_20240721': 2,
              'evaluation_swiss-experiments_2old_20240721': 2,
              'evaluation_swiss-experiments_10_20240805': 10,
              'evaluation_swiss-experiments_final_20240810': 900,
              'evaluation_swiss-experiments_8_20240728': 8,
              'evaluation_swiss-experiments_3_20240812': 302,
              'evaluation_swiss-experiments_9_20240804': 9,
              'evaluation_swiss-experiments_18_20240814': 18,
              'evaluation_swiss-experiments_3_20240723': 3,
              'evaluation_swiss-experiments_3oldr4_20241001': 3004,
              'evaluation_swiss-experiments_3old_20240723': 3,
              'evaluation_swiss-experiments_16_20240803': 16,
              'evaluation_swiss-experiments_11_20240806': 11,
              'evaluation_swiss-experiments_6_20240730': 6,
              'evaluation_swiss-experiments_12_20240808': 12,
              'evaluation_swiss-experiments_finalv2_20240811': 902,
              'evaluation_swiss-experiments_17_20240802': 17,
              'evaluation_swiss-experiments_7_20240727': 7,
              'evaluation_swiss-experiments_13_20240809': 13,
              'evaluation_swiss-experiments_finalv3_20240817': 903,
              'evaluation_swiss-experiments_finalv4_20240818': 904,
              'evaluation_swiss-experiments_2_20240816': 202,
              'evaluation_swiss-experiments_1_20240819': 102,
              'evaluation_swiss-experiments_finalv22_20240824': 9022,
              'evaluation_swiss-experiments_finalv42_20240827': 9042,
              'evaluation_swiss-experiments_122_20240826': 122,
              'evaluation_swiss-experiments_final6_20240901': 906,
              'evaluation_swiss-experiments_21_20240831': 21,
              'evaluation_swiss-experiments_finalv5_20240829': 905,
              'evaluation_swiss-experiments_finalv32_20240902': 9032,
              'evaluation_swiss-experiments_22_20240904': 22,
              'evaluation_swiss-experiments_23_20240905': 23,
              'evaluation_swiss-experiments_24_20240914': 24,
              'evaluation_swiss-experiments_final7_20240928': 907,
              'evaluation_swiss-experiments_25_20241002': 25,
              'evaluation_swiss-experiments_27_20241003': 27,
              'evaluation_swiss-experiments_26_20241002': 26,
              'evaluation_swiss-experiments_29_20241005': 29,
              'evaluation_swiss-experiments_28_20241002': 28,
              # 'evaluation_swiss-experiments_final8_XXXXXXXX': 908,
              }


def read_configs(fp, args=['data_dir', 'processed_dir', 'in_channels', 'nir_drop_chance', 'rescale', 'normalize',
                           'normalize_by_dataset', 'normalize_by_imagenet', 'val_metric',
                           'loss_function_energy', "weighted_sampling", 'auto_resample', 'apply_count_weights',
                           'apply_edge_weights',
                           'model_type', 'backbone', 'backbone_warmup', 'load', 'pretrained_ckpt', 'reset_head',
                           'warmup', 'n_energy_bins',
                           'band_sequence', 'epochs', 'lr', 'path_size', 'batch_size', 'batch_size_val',
                           'checkpoint_per_epoch'], ind_test=False):
    """read configuration file .json"""

    # Get the folder name as the model name
    model_name = '_'.join(os.path.basename(os.path.dirname(fp)).split('_')[1:-4])

    with open(fp) as f:
        config_train = json.loads(f.read())

    # add config file to the args
    config_train['config_file'] = fp
    args.append('config_file')

    # add best model to the args
    config_train['best_model'] = os.path.join(os.path.dirname(fp), 'model', 'BestModel.pth') if os.path.exists(
        os.path.join(os.path.dirname(fp), 'model', 'BestModel.pth')) else None
    args.append('best_model')

    # get the creation data of the config file
    if ind_test:
        # for independent test, the model creation date is extracted from the load path
        load_path = config_train['load']
        # get the parent folder of the load path
        load_path = os.path.dirname(load_path)
        config_train['creation_date'] = os.path.basename(os.path.dirname(load_path)).split('_')[-4]
    else:
        config_train['creation_date'] = os.path.basename(os.path.dirname(fp)).split('_')[-4]
    # Change the date format from english words to numbers
    config_train['creation_date'] = config_train['creation_date'].replace('Jan', '01').replace('Feb', '02').replace(
        'Mar', '03').replace('Apr', '04').replace('May', '05').replace('Jun', '06').replace('Jul', '07').replace(
        'Aug', '08').replace('Sep', '09').replace('Oct', '10').replace('Nov', '11').replace('Dec', '12')
    config_train['creation_date'] = '2024' + config_train['creation_date']
    args.append('creation_date')

    config_train['creation_time'] = os.path.basename(os.path.dirname(fp)).split('_')[-3]
    # change the time format from h-m-s to hms
    config_train['creation_time'] = config_train['creation_time'].replace('-', '')
    args.append('creation_time')

    return model_name, dict([(i, config_train.get(i, None)) for i in args])


def get_test_metrics(fp):
    """read test metrics from the last few lines of a log txt file"""

    with open(fp) as f:
        lines = f.readlines()
        # start to search from the second place where 'INFO: Network:' appears, and search from the bottom
        for i in range(len(lines) - 1, 0, -1):
            if 'INFO: Network:' in lines[i]:
                break
        lines = lines[i:]

        # start from where empty line is found
        for i in range(1, len(lines)):
            if lines[i] == '\n':
                break
        lines = lines[i:]
        # get the test metrics
        test_metrics = {}
        for idx, line in enumerate(lines):
            if 'INFO: ' in line:
                line = line.split('INFO: ')[1]
                key, value = line.split(': ')
                if 'loss' in key or 'watershed_cumprod' in key:
                    test_metrics[key] = value.split('\n')[0]
                # break until the first 'confusion matrix'
                if 'confusion_matrix/watershed_cumprod' in key:
                    # add the line after the current one
                    test_metrics[key] = value + lines[idx + 1]
                    # retrieve numbers from a string using regex
                    if 'device' in test_metrics[key]:
                        cm = re.findall(r'\d+', test_metrics[key])[:-1]
                    else:
                        cm = re.findall(r'\d+', test_metrics[key])
                    test_metrics[key] = '_'.join(cm)
                    break

        return test_metrics


def check_paused(fp):
    """check if the training was paused by checking if 'test' is in the second to the last line in the logs.txt file"""
    with open(fp) as f:
        lines = f.readlines()
        if 'test' in lines[-2]:
            # searching from the bottom of the logs.text file 'INFO: Starting epoch 1 ...' and extract the epoch number
            return False
        else:
            with open(fp) as f:
                lines = f.readlines()
                for i in range(len(lines) - 1, 0, -1):
                    if 'INFO: Starting epoch' in lines[i]:
                        # return the number in the line
                        return int(lines[i].split(' ')[-2])


def batch_run(wd, path_filter=None, prefix='run', out_fp='./tmp/configs.csv', evaluation_metrics=False, ind_test=False):
    # get all folders starting with run in the working directory
    folder_list = [i for i in os.listdir(wd) if i.startswith(prefix)]
    if path_filter is not None:
        folder_list = [fp for fp in folder_list if path_filter in fp]

    fps = [os.path.join(wd, i, 'config.json') for i in folder_list]

    # get all paused folders
    paused_stats = [check_paused(os.path.join(wd, i, 'logs.txt')) for i in folder_list]
    non_paused_folders = [i for idx, i in enumerate(folder_list) if paused_stats[idx] is False]

    # read all configurations
    model_names, configs = zip(*[read_configs(fp, ind_test=ind_test) for fp in fps])

    # add paused status to the configurations
    for i, paused in enumerate(paused_stats):
        configs[i]['paused'] = paused

    if evaluation_metrics:
        for i in range(len(configs)):
            configs[i]['test_metrics'] = get_test_metrics(os.path.join(wd, folder_list[i], 'logs.txt')) if folder_list[i] in non_paused_folders else None

    # Zip the model names and configurations and sort by creation date
    zipped = list(zip(model_names, configs))
    zipped.sort(key=lambda x: x[1]['creation_date'])
    model_names, configs = zip(*zipped)

    # put all configurations in a dictionary
    # solve the problem of the same model name
    configs_dict = {}
    for i, model_name in enumerate(model_names):
        if model_name in configs_dict:
            model_name = model_name + '_' + str(i)
        configs_dict[model_name] = configs[i]

    # turn into a csv file
    df = pd.DataFrame(configs_dict)
    df = df.T
    # sort by creation date
    df = df.sort_values(by=['creation_date', 'creation_time'], ascending=False)
    # move creation date and time to the first column
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]

    # Use index as the first column and rename to run_id
    df = df.reset_index()
    df = df.rename(columns={'index': 'run_id'})

    # test and evaluation only when 'in_channels', 'auto_resample', 'normalize', 'normalize_by_dataset', 'normalize_by_imagenet', is the same as in training
    same_params = ['in_channels', 'normalize', 'normalize_by_dataset', 'normalize_by_imagenet', 'model_type',
                   'band_sequence']

    # # Convert dictionary output of get_eval_logs to a dataframe
    # eval_logs = get_eval_logs(wd)
    # df_eval = pd.DataFrame(eval_logs)
    # # Merge the two dataframes using load as the key
    # df = df.merge(df_eval, left_on='best_model', right_on='load', how='left', copy=False)
    # # check if value in same_paras the same in df and df_eval
    # for i in same_params:
    #     if i != 'auto_resample':
    #         df['valid_val'] = df[i + '_x'] == df[i + '_y'] if df[i + '_x'] is not None else True
    #
    # # the logs for test was saved independently instead...
    # # get test metrics for unfinished runs
    # test_logs = get_test_logs(wd)
    # df_test = pd.DataFrame(test_logs)
    # # Merge the two dataframes using load as the key
    # df = df.merge(df_test, left_on='best_model', right_on='load', how='left', copy=False)
    # # check if value in same_paras the same in df and df_eval
    # for i in same_params:
    #     df['valid_test'] = df[i + '_x'] == df[i] if df[i + '_x'] is not None else True
    #
    # # overwrite test_metrics from df_test on df and remove test_metrics_x and test_metrics_y
    # df['test_metrics'] = df['test_metrics_y'].combine_first(df['test_metrics_x'])
    # df = df.drop(columns=['test_metrics_x', 'test_metrics_y'])
    # # drop columns in same_params
    # same_params.extend(['config_file', 'load'])
    # df = df.drop(columns=[i for i in same_params] + [i + '_y' for i in same_params])

    # change 'auto_resample' to 'auto_resample_test'
    df = df.rename(columns={'auto_resample': 'auto_resample_test'})
    # remove '_x' in the column names and change '_y' to '_val'
    df.columns = [i.replace('_x', '') for i in df.columns]
    df.columns = [i.replace('_y', '_val') for i in df.columns]

    # move test_metrics to the 5th columns
    cols = df.columns.tolist()
    cols = cols[:4] + cols[-1:] + cols[4:-1]
    df = df[cols]

    # save the csv file
    df.to_csv(out_fp, index=False)
    return df


def get_eval_logs(logs_folder,
                  args=['load', 'report_folder', 'in_channels', 'auto_resample', 'normalize', 'normalize_by_dataset',
                        'normalize_by_imagenet', 'evaluate_datasets', 'config_file', 'model_type', 'band_sequence']):
    """Get the folder for evaluation reports"""
    # Read through the folder starting with evaluation in the folder name
    folder_list = [i for i in os.listdir(logs_folder) if i.startswith('evaluation')]
    fps = [os.path.join(logs_folder, i, 'config.json') for i in folder_list]

    # Extract the load parameter and the report_folder in the config file in the folder
    args_list = []
    for fp in fps:
        with open(fp) as f:
            config = json.loads(f.read())
            params = dict([(i, config.get(i, None)) for i in args])
            # add folder name to the args
            params['eval_run_folder'] = os.path.basename(os.path.dirname(fp))
            args_list.append(params)

    return args_list


def get_test_logs(logs_folder, args=['load', 'in_channels', 'auto_resample', 'normalize', 'normalize_by_dataset',
                                     'normalize_by_imagenet', 'config_file', 'model_type', 'band_sequence']):
    """Get the test metrics for paused runs"""
    # Read through the folder starting with evaluation in the folder name
    folder_list = [i for i in os.listdir(logs_folder) if i.startswith('test')]
    fps = [os.path.join(logs_folder, i, 'config.json') for i in folder_list]

    # Extract the load parameter and the report_folder in the config file in the folder
    args_list = []
    for fp in fps:
        with open(fp) as f:
            config = json.loads(f.read())
            params = dict([(i, config.get(i, None)) for i in args])
            # add folder name to the args
            params['test_run_folder'] = os.path.basename(os.path.dirname(fp))
            params['test_metrics'] = get_test_metrics(
                os.path.join(logs_folder, os.path.basename(os.path.dirname(fp)), 'logs.txt'))
            args_list.append(params)

    return args_list


def extract_test_metrics(in_fp='./tmp/configs.csv', out_fp='./tmp/eval_metrics.csv', run_id_key='swiss'):
    """Extract test metrics from configs.csv and save to a new csv file"""

    scenario_dict = {
        "swiss-experiments_baseline_20240719": "baseline (5 energy bins)",
        "swiss-experiments_0_20240719": "baseline (5 energy bins)",
        "swiss-experiments_1_20240720": "transformer (with size weights)",
        "swiss-experiments_2_20240721": "pretrained imagenet backbone (with size weights)",
        "swiss-experiments_3_20240723": "pretrained tree model backbone (with size weights)",
        "swiss-experiments_3oldr4_20241001": "3oldr4",
        "swiss-experiments_4_20240724": "with resolution information",
        "swiss-experiments_5_20240725": "with size weights",
        "swiss-experiments_6_20240730": "4 bands",
        "swiss-experiments_7_20240727": "tv46",
        "swiss-experiments_8_20240728": "focal loss",
        "swiss-experiments_9_20240804": "MANet with resolution information",
        "swiss-experiments_10_20240805": "resnet101",
        "swiss-experiments_11_20240806": "weighted sampler",
        "swiss-experiments_12_20240808": "loss_count_cumprod/val",
        "swiss-experiments_13_20240809": "no rescale",
        "swiss-experiments_14_20240731": "no normalization",
        "swiss-experiments_16_20240803": "10 energy bins",
        "swiss-experiments_17_20240802": "3 energy bins",
        "swiss-experiments_18_20240814": "train transformer decoder only",
        "swiss-experiments_final_20240810": "baseline + others",
        "swiss-experiments_finalv2_20240811": "final with loss_comb",
        "swiss-experiments_finalv3_20240817": "final with no size weights with loss_comb",
        "swiss-experiments_finalv4_20240818": "final with weighted-sampling with size weights with loss_comb",
        'swiss-experiments_2_20240816': "pretrained imagenet backbone",
        'swiss-experiments_1_20240819': "train transformer encoder and decoder",
        'swiss-experiments_3_20240812': "pretrained tree model backbone",
        'swiss-experiments_finalv22_20240824': "finalv2 with loss_comb (corrected)",
        'swiss-experiments_finalv42_20240827': "finalv4 with loss_comb (corrected)",
        'swiss-experiments_122_20240826': "run_id 12 with loss_count_cumprod (corrected)",
        'swiss-experiments_final6_20240901': "final with loss_count_cumprod (corrected)",
        'swiss-experiments_21_20240831': "loss_comb (corrected)",
        'swiss-experiments_finalv5_20240829': "finalv2 with unet_with_scalar_v2",
        'swiss-experiments_finalv32_20240902': 'final with no size weights with loss_comb (corrected)',
        'swiss-experiments_22_20240904': 'normalize by dataset',
        'swiss-experiments_23_20240905': 'mixed vision transformer with encoder freeze 100 epochs and train all for 400 epochs',
        'swiss-experiments_24_20240914': 'mixed vision transformer (decoder only) with resolution information',
        'swiss-experiments_final7_20240928': "final with loss_energy",
        #'swiss-experiments_final8_XXXXXXXX': "fnalv7 with no backbone warmup",
        'swiss-experiments_25_20241002': "resolution information + size weights",
        'swiss-experiments_26_20241002': "transformer encoder freeze + res + size weights",
        'swiss-experiments_27_20241003': "transformer encoder freeze + size weights",
        'swiss-experiments_28_20241002': "pretrained tree + res",
        'swiss-experiments_29_20241003': "loss/val and sobel * 100",
        'swiss-experiments_1long_20241005': "transformer encoder and decoder trained for 1000 epochs",
    }

    # convert dictionary to a dataframe
    df_scenario = pd.DataFrame.from_dict(scenario_dict, orient='index', columns=['scenario'])
    df_scenario = df_scenario.reset_index()
    df_scenario = df_scenario.rename(columns={'index': 'run_id'})

    # Extract the experiment name and metrics to a new csv file
    df = pd.read_csv(in_fp)

    if run_id_key is not None:
        df = df[df['run_id'].isin([i for i in df['run_id'].astype(str).values if run_id_key in i])][
            ['run_id', 'creation_date', 'test_metrics']]
    else:
        df = df[['run_id', 'creation_date', 'test_metrics']]

    # drop duplicates for cases with evaluations
    df = df.drop_duplicates(subset='run_id', keep='first')
    # drop rows with NaN values
    df = df.dropna()
    df.reset_index(inplace=True)

    # convert the dictionary column named 'test_metrics' to a dataframe
    dict_ = {'_'.join([df['run_id'][i].split('_')[0], str(df['run_id'][i].split('_')[1]), str(df['creation_date'][i])]): ast.literal_eval(df['test_metrics'][i]) for i in range(0, len(df))}
    df = pd.DataFrame.from_dict(dict_).T
    # reset index and name as run_id
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'run_id'})

    # merge the two dataframes on run_id
    df = df_scenario.merge(df, left_on='run_id', right_on='run_id', how='right', copy=False)

    df.dropna(subset=['loss/test'], inplace=True)

    # Split confusion matrix column into separate columns by seperateor '_'
    cm_col = [i for i in df.columns if 'confusion_matrix' in i][0]
    df[['T_0', 'F_0', 'F_1', 'T_1']] = df[cm_col].str.split('_').apply(pd.Series)
    # convert a column of list into multiple columns
    df[['T_0', 'F_0', 'F_1', 'T_1']] = df[['T_0', 'F_0', 'F_1', 'T_1']].astype(int)
    df['T_0_'] = df['T_0'] / (df['T_0'] + df['F_0'])
    df['T_1_'] = df['T_1'] / (df['T_1'] + df['F_1'])
    df['F_0_'] = df['F_0'] / (df['T_0'] + df['F_0'])
    df['F_1_'] = df['F_1'] / (df['T_1'] + df['F_1'])

    # recalculate loss_count in percentages
    # 528 is the number of images in the test dataset, 2939 is the number of instances in the test dataset 6 is the number of resampled patches
    df['rMAE(%)'] = df['loss_count_cumprod/test'].astype(float) * 528 * 6 / (2939 * 6) * 100
    df['rMAE(%)_ORI'] = df['loss_count_cumprod_10/test'].astype(float) * 528 / 2939 * 100

    # sort by loss_count_cumprod/test from small to big
    df = df.sort_values(by=['iou/watershed_cumprod/test/1'], ascending=False)

    df.to_csv(out_fp, index=False)
    return df


def extract_ap50(in_fp, out_fp):
    """Extract ap50 metrics from evaluation logs and save to a csv file"""


    # when auto resample is True in evaluation
    name_id_map = {'val dataset (non-resampled)': 0,
                   'large objects (> 389 pixels) in val dataset (non-resampled)': 1,
                   'medium objects (35 - 389 pixels) val dataset (non-resampled)': 2,
                   'small objects (< 35 pixels) val dataset (non-resampled)': 3,
                   'val in CH at resolution 0.11999999731779099 (scale 1.0)': 4,
                   'val in CH at resolution 0.25 (scale 1.0)': 5,

                   'test dataset (non-resampled)': 6,
                   'large objects (> 389 pixels) in test dataset (non-resampled)': 7,
                   'medium objects (35 - 389 pixels) test dataset (non-resampled)': 8,
                   'small objects (< 35 pixels) test dataset (non-resampled)': 9,
                   'test in CH at resolution 0.11999999731779099 (scale 1.0)': 10,
                   'test in CH at resolution 0.25 (scale 1.0)': 11,

                   'val in CH at resolution 0.6000000238418579 (scale 0.41999998688697815)': 12,
                   'val in CH at resolution 0.30000001192092896 (scale 0.8299999833106995)': 13,
                   'val in CH at resolution 0.4000000059604645 (scale 0.6200000047683716)': 14,
                   'val in CH at resolution 0.20000000298023224 (scale 1.25)': 15,
                   'val in CH at resolution 0.20000000298023224 (scale 0.6000000238418579)': 16,
                   'val in CH at resolution 0.6000000238418579 (scale 0.20000000298023224)': 17,
                   'val in CH at resolution 0.5 (scale 0.23999999463558197)': 18,
                   'val in CH at resolution 0.30000001192092896 (scale 0.4000000059604645)': 19,
                   'val in CH at resolution 0.5 (scale 0.5)': 20,
                   'val in CH at resolution 0.4000000059604645 (scale 0.30000001192092896)': 21,

                   'test in CH at resolution 0.6000000238418579 (scale 0.41999998688697815)': 22,
                   'test in CH at resolution 0.30000001192092896 (scale 0.8299999833106995)': 23,
                   'test in CH at resolution 0.4000000059604645 (scale 0.6200000047683716)': 24,
                   'test in CH at resolution 0.20000000298023224 (scale 1.25)': 25,
                   'test in CH at resolution 0.20000000298023224 (scale 0.6000000238418579)': 26,
                   'test in CH at resolution 0.6000000238418579 (scale 0.20000000298023224)': 27,
                   'test in CH at resolution 0.5 (scale 0.23999999463558197)': 28,
                   'test in CH at resolution 0.30000001192092896 (scale 0.4000000059604645)': 29,
                   'test in CH at resolution 0.5 (scale 0.5)': 30,
                   'test in CH at resolution 0.4000000059604645 (scale 0.30000001192092896)': 31,

                   'val dataset (resampled and non-resampled)': 32,
                   'large objects (> 389 pixels) in val dataset (resampled and non-resampled)': 33,
                   'medium objects (35 - 389 pixels) val dataset (resampled and non-resampled)': 34,
                   'small objects (< 35 pixels) val dataset (resampled and non-resampled)': 35,

                   'test dataset (resampled and non-resampled)': 36,
                   'large objects (> 389 pixels) in test dataset (resampled and non-resampled)': 37,
                   'medium objects (35 - 389 pixels) test dataset (resampled and non-resampled)': 38,
                   'small objects (< 35 pixels) test dataset (resampled and non-resampled)': 39,

                   }

    with open(in_fp) as f:
        lines = f.readlines()

        # get model name (run_id) for evaluation from the first line, where starts "evaluation_swiss-experiments" plus the string the '_'
        run_id = ''.join(
            ['evaluation_swiss-experiments_', lines[0].split('evaluation_swiss-experiments_')[1].split('_')[0]])

        # split the line 40 by '_' and get the index of the element contains swiss-experiments and get the element two times after it
        date = lines[39].split('_')[[i for i, j in enumerate(lines[0].split('_')) if 'swiss-experiments' in j][0] + 2]
        # convert Aug18 to 20240818
        date = date.replace('Aug', '08').replace('Sep', '09').replace('Oct', '10').replace('Nov', '11').replace('Dec',
                                                                                                                '12').replace('Jul', '07')
        model_creation_date = '2024' + date

        # get the folder path from the line starting with "reading df from"
        processed_data = re.findall(r'Reading df from (.+)', lines[2])[0]
        processed_data = os.path.basename(os.path.dirname(processed_data))

        # start to search from the second place where 'INFO: Network:' appears, and search from the bottom
        for i in range(len(lines) - 1, 0, -1):
            if 'INFO: Not extracting images.' in lines[i]:
                break
        lines = lines[i + 2:]

        # Get iou threshold from the first line
        try:
            iou_thresh = lines[0].split('IOU = ')[-1].split('\n')[0]

            name = []
            ap = []
            tp = []

            for l in lines:
                if 'INFO: ' in l:
                    line = l.split('INFO: ')[1]
                    if 'AP' in line:
                        if 'AP50 for' in line:
                            line = line.split('AP50 for ')[-1]
                        elif 'AP for' in line:
                            line = line.split('AP for ')[-1]
                        line = line.replace('scale:', 'scale')
                        name.append(line.split(': ')[0])
                        ap.append(line.split(': ')[-1].split('\n')[0])
                    if 'TP' in line:
                        line = line.split('TP for ')[-1]
                        line = line.replace('scale:', 'scale')
                        tp.append(line.split(': ')[-1].split('\n')[0])

            df = pd.DataFrame({'run_id': [run_id] * len(ap), 'model_creation_date': [model_creation_date] * len(ap),
                               'processed_data': [processed_data] * len(ap), 'iou_thresh': [iou_thresh] * len(ap),
                               'name': name, 'ap': ap, 'tp': tp})
            # ids, uniques = pd.factorize(df['name'])
            # map the name to name_id
            df['name_id'] = df['name'].map(name_id_map).astype(int)

            # set the data types
            df['ap'] = df['ap'].astype(float)
            df['tp'] = df['ap'].astype(int)
            df['iou_thresh'] = df['iou_thresh'].astype(float)

            # replace baseline in the string of run_id with 0
            df['run_id'] = df['run_id'].str.replace('baseline', '0')

            if out_fp is not None:
                if os.path.exists(out_fp):
                    df.to_csv(out_fp, mode='a', header=False, index=False)
                else:
                    df.to_csv(out_fp, index=False)
            return df
        except:
            pass


def viz_ap50(fp, out_fp, dataset=['test'], name_ids=list(range(6, 12)) + list(range(22, 32)), run_id_list=None, iou_thresh_list=None):
    """Visualize the ap50 metrics from the csv file"""

    df = pd.read_csv(fp)

    df = pd.concat([df[df['name'].str.contains(i)] for i in dataset])
    df = df[df['name_id'].isin(name_ids)] if name_ids is not None else df

    df['run_id'] = df['run_id'] + '_' + df['model_creation_date'].astype(str)
    # ids, uniques = pd.factorize(df['run_id'])
    # run_id_map = dict(zip(uniques, range(len(uniques))))
    df['run_id'] = df['run_id'].map(run_id_map).astype(int)
    df = df[df['run_id'].isin(run_id_list)] if run_id_list is not None else df
    # set nan in column ap to 0
    df['ap'] = df['ap'].fillna(0)

    # set data types
    df['run_id'] = df['run_id'].astype(int)
    df['processed_data'] = df['processed_data'].astype(str)
    df['iou_thresh'] = df['iou_thresh'].astype(float)
    df['name'] = df['name'].astype(str)
    df['ap'] = df['ap'].astype(float)
    df['name_id'] = df['name_id'].astype(int)

    df = df[df['iou_thresh'].isin(iou_thresh_list)] if iou_thresh_list is not None else df

    # for each name_id create a bar chart independently, and group them into a grid
    fig, ax = plt.subplots(int(len(name_ids) / 2), 2)

    out_dict = {}
    run_ids = sorted(df['run_id'].unique())

    name_id_idx = 0
    for r, c in [(i, j) for i in range(int(len(name_ids) / 2)) for j in range(2)]:
        if name_id_idx > len(name_ids):
            break
        data = df[df['name_id'] == name_ids[name_id_idx]]
        sns.barplot(x='run_id', y='ap', hue='iou_thresh', data=data, ax=ax[r][c])

        # add text using name_id on the top left corner of each subplot
        ax[r][c].text(0.05, 0.95, f'({chr(64 + name_id_idx + 1).lower()})', transform=ax[r][c].transAxes, fontsize=5, verticalalignment='top')

        # set x label for the last two only
        if r == int(len(name_ids) / 2) - 1:
            ax[r][c].set_xlabel('Model ID', fontdict={'fontsize': 4})
        else:
            ax[r][c].set_xlabel('')
        # rotate x labels
        ax[r][c].tick_params(axis='x', rotation=90)
        # set the distance of x label from the x axis
        ax[r][c].xaxis.labelpad = 0.4
        # # remove x tick labels if not the last row
        # if r != int(len(name_ids) / 2) - 1:
        #     ax[r][c].set_xticklabels([])
        # remove x ticks and set the font for tick labels
        ax[r][c].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, labelsize=4)
        # set the distance of x tick labels from the x axis
        ax[r][c].tick_params(axis='x', which='major', pad=0.4)

        # set y label for the first and last one in the first column
        if (c == 0 and r == 0) or (c == 0 and r == int(len(name_ids) / 2) - 1):
            # ax[r][c].set_ylabel('Average Precision (AP)')
            ax[r][c].set_ylabel('')
        else:
            ax[r][c].set_ylabel('')
        # y limit from 0 to 1
        ax[r][c].set_ylim(0, 1.1)
        # set major ticks with labels for the first column and set the labels fontsize as 5
        if c == 0:
            ax[r][c].set_yticks([0.2, 0.5, 0.8, 1])
            ax[r][c].set_yticklabels([0.2, 0.5, 0.8, 1], fontsize=5)
        else:
            ax[r][c].set_yticks([0.2, 0.5, 0.8, 1])
            ax[r][c].set_yticklabels([])
        # remove minor ticks and set the font size of tick labels to 5
        ax[r][c].tick_params(axis='y', which='minor', left=False, right=False)

        # put a start on the highest bar per iou group
        # Retrieve the patches (bars) and the legend from the plot
        patches = ax[r][c].patches
        handles, labels = ax[r][c].get_legend_handles_labels()
        # Create a dictionary to map legend texts to their colors
        hue_color_map = {float(label): handle.get_facecolor() for handle, label in zip(handles, labels)}

        # Function to select bars by hue
        def select_bars_by_hue(hue):
            selected_bars = []
            target_color = hue_color_map[hue]
            for bar in patches:
                if tuple(bar.get_facecolor()) == tuple(target_color):  # Match the RGBA color
                    selected_bars.append(bar)
            return selected_bars

        count_1 = {i: len(data['run_id'].unique()) * [0] for i in iou_thresh_list}
        count_2 = {i: len(data['run_id'].unique()) * [0] for i in iou_thresh_list}
        count_3 = {i: len(data['run_id'].unique()) * [0] for i in iou_thresh_list}

        # highlight the highest value in each group with a star
        for iou in data['iou_thresh'].unique():
            # Example: Select and manipulate bars for 'Type 1'
            type_1_bars = select_bars_by_hue(iou)
            # Get the heights of the bars
            bar_heights = [bar.get_height() for bar in type_1_bars]
            # get the index of the highest value in bar_heights
            max_height_idx_list = [i for i, j in enumerate(bar_heights) if j == max(bar_heights)]
            for max_height_idx in max_height_idx_list:
                # get the bar with the highest value
                max_height_bar = type_1_bars[max_height_idx]
                # add a star in the middle of the bar
                ax[r][c].annotate('*', (
                    max_height_bar.get_x() + max_height_bar.get_width() / 2 + 0.01, max_height_bar.get_height()), ha='center',
                                  va='center', fontsize=5, color='red')
            count_1.update({iou: dict(zip(run_ids,
                                          [count_1[iou][i] + 1 if i in max_height_idx_list else count_1[iou][i] for i in
                                           range(len(count_1[iou]))]))})

            # highlight the second hightest value in each group
            # remove the highest value
            bar_heights_ = [item for index, item in enumerate(bar_heights) if index not in max_height_idx_list]
            # get the index of the second highest value in bar_heights
            second_max_height_idx_list = [i for i, j in enumerate(bar_heights) if j == max(bar_heights_)]
            for second_max_height_idx in second_max_height_idx_list:
                # get the bar with the second highest value
                second_max_height_bar = type_1_bars[second_max_height_idx]
                # add a red triangle in the middle of the bar
                # move up a bit the triangle if it overlaps with the star
                offset2 = 0.05
                if second_max_height_bar.get_height() == max_height_bar.get_height() and second_max_height_bar.get_x() == max_height_bar.get_x():
                    offset2 += 0.05
                ax[r][c].annotate(r'$\blacktriangle$', (
                    second_max_height_bar.get_x() + second_max_height_bar.get_width() / 2 + 0.03,
                    second_max_height_bar.get_height() + offset2), ha='center', va='center', fontsize=2, color='green')
            count_2.update({iou: dict(zip(run_ids,
                                          [count_2[iou][i] + 1 if i in second_max_height_idx_list else count_2[iou][i]
                                           for i in range(len(count_2[iou]))]))})

            # highlight the third highest value in each group
            # remove the second highest value
            bar_heights_ = [item for index, item in enumerate(bar_heights) if
                            index not in max_height_idx_list + second_max_height_idx_list]
            # get the index of the third highest value in bar_heights
            third_max_height_idx_list = [i for i, j in enumerate(bar_heights) if j == max(bar_heights_)]
            for third_max_height_idx in third_max_height_idx_list:
                # get the bar with the third highest value
                third_max_height_bar = type_1_bars[third_max_height_idx]
                # add a red cross in the middle of the bar
                # move the cross up a bit if it overlaps with the star on the highest bar
                offset3 = 0.05
                if third_max_height_bar.get_height() == max_height_bar.get_height() and third_max_height_bar.get_x() == max_height_bar.get_x():
                    offset3 += 0.05
                if third_max_height_bar.get_height() + offset3 == second_max_height_bar.get_height() + offset2 and third_max_height_bar.get_x() == second_max_height_bar.get_x():
                    offset3 += 0.05
                ax[r][c].annotate('+', (third_max_height_bar.get_x() + third_max_height_bar.get_width() / 2 + 0.02,
                                        third_max_height_bar.get_height() + offset3), ha='center', va='center',
                                  fontsize=3, color='blue')
            count_3.update({iou: dict(zip(run_ids,
                                          [count_3[iou][i] + 1 if i in third_max_height_idx_list else count_3[iou][i]
                                           for i in range(len(count_3[iou]))]))})

        # add a thin gray line when y=1
        ax[r][c].axhline(y=1, color='gray', linewidth=0.3, linestyle='-')

        # remove legends
        ax[r][c].get_legend().remove()

        # Save the count of the highest, second highest, and third highest values in each group
        out_dict[name_id_idx] = {'first': count_1, 'second': count_2, 'third': count_3}

        name_id_idx += 1

    # export statistics based on the count of highest, second highest, and third highest values in each group
    # convert nested out_dict to a dataframe
    out_df = pd.DataFrame.from_dict(
        {(i, j, k): out_dict[i][j][k] for i in out_dict.keys() for j in out_dict[i].keys() for k in
         out_dict[i][j].keys()}, orient='index')
    out_df = out_df.reset_index()
    out_df = out_df.rename(columns={'level_0': 'name_id', 'level_1': 'rank', 'level_2': 'iou'})
    out_df = out_df.melt(id_vars=['name_id', 'rank', 'iou'], var_name='run_id', value_name='count')

    def f(stats, col):
        # get all run_id for the maximum count
        aa = stats.loc[stats.groupby([col])['count'].idxmax()]
        # merge rest_test and aa on name_id
        bb = stats.merge(aa, on=col, how='left', suffixes=('', '_val'))
        cc = bb.loc[bb['count_val'] == bb['count'], [col, 'run_id', 'count']]
        dd = {i: cc[cc[col] == i]['run_id'].values for i in cc[col].unique()}
        return dd

    # test_mask = out_df['name_id'] > 5
    # # all
    # stats = out_df.loc[test_mask].groupby(['run_id'])['count'].sum().reset_index()
    # stats.to_csv('./tmp/test_all_count.csv', index=False)
    # # group by name_id
    # stats = out_df.loc[test_mask].groupby(['name_id', 'run_id'])['count'].sum().reset_index()
    # stats.to_csv('./tmp/test_count_by_nameid.csv', index=False)
    # print("Group by name_id:\n", f(stats, 'name_id'))
    # # group by iou
    # stats = out_df.loc[test_mask].groupby(['iou', 'run_id'])['count'].sum().reset_index()
    # stats.to_csv('./tmp/test_count_by_iou.csv', index=False)
    # print("Group by iou:\n", f(stats, 'iou'))
    # # group by name id and iou
    # stats = out_df.loc[test_mask].groupby(['name_id', 'iou', 'run_id'])['count'].sum().reset_index()
    # stats.to_csv('./tmp/test_count_by_nameid_iou.csv', index=False)
    # print("Group by name_id and iou:\n", f(stats))
    # # group by name id, low ious
    # stats = out_df.loc[test_mask & (out_df['iou'].isin([0.3, 0.5]))].groupby(['name_id', 'run_id'])['count'].sum().reset_index()
    # stats.to_csv('./tmp/test_count_by_nameid_lowious.csv', index=False)
    # # print("Group by name_id with low ious:\n", f(stats))
    # # group by name id, high ious
    # stats = out_df.loc[test_mask & (out_df['iou'].isin([0.6, 0.9]))].groupby(['name_id', 'run_id'])['count'].sum().reset_index()
    # stats.to_csv('./tmp/test_count_by_nameid_lowious.csv', index=False)
    # # print("Group by name_id with high ious:\n", f(stats))

    # adjust the margin between subplots
    plt.subplots_adjust(hspace=0.17, wspace=0.02)

    # add one legend on the bottom of the figure, arrange horizontally
    handles, labels = ax[0][0].get_legend_handles_labels()
    # combine it with another legend for annotations as symbol
    handles.append(plt.Line2D([0], [0], marker='*', color='red', markerfacecolor='red', markersize=3, linestyle='None',
                              linewidth=0.1))
    handles.append(plt.Line2D([0], [0], marker=r'$\blacktriangle$', color='green', markerfacecolor='green', markersize=3,
                              linestyle='None'))
    handles.append(plt.Line2D([0], [0], marker='+', color='blue', markerfacecolor='blue', markersize=3, linestyle='None'))
    labels.extend(['Highest', 'Second Highest', 'Third Highest'])
    # add the legend with minimum margin between handles
    plt.figlegend(handles, labels, loc='lower center', ncol=7, fontsize=5, borderaxespad=0.1, borderpad=0.1,
                  handletextpad=0.1, handlelength=1, columnspacing=0.5)
    # reduce the margin between the legend and the subplots
    plt.subplots_adjust(bottom=0.08)

    # plt.tight_layout()
    plt.savefig(out_fp, dpi=600)


def load_and_prepare_data(gpkg_file_path):
    # Read the gpkg file as a dataframe named df
    df = gpd.read_file(gpkg_file_path)

    # Extract the basename and split by "_", then get the last piece as iou_thresh
    base_name = os.path.basename(gpkg_file_path)
    file_name = os.path.splitext(base_name)[0]
    iou_thresh = file_name.split('_')[-1]

    # Extract the folder name and split by "_", the first part is run_id
    folder_name = os.path.basename(os.path.dirname(gpkg_file_path))
    run_id = folder_name.split('_')[0]

    # Create filters for resampled and non-resampled ones or all in one using column named resampled in df
    non_resampled_filter = df['resampled'] == 0

    # Create filters for test set using a column named Class in df
    test_filter = df['Class'] == 'test'

    # Create filters to pick rows with Label > 0 in df
    label_filter = df['Label'] > 0

    # Create a new df named data with only non-resampled ones for the test set and Label > 0
    data = df[non_resampled_filter & test_filter & label_filter].copy()

    # Add run_id and iou_thresh to the data
    data['run_id'] = run_id
    data['iou_thresh'] = iou_thresh

    # remove iou_thresh == test
    data = data[data['iou_thresh'] != 'test']
    return data


# def visualize_multiple_patch_ap(gpkg_file_paths, scenario='a', out_fp='./tmp/swiss_patch_ap50.png'):
#
#     run_id_map = {'v' + k.split('_')[-1]: v for k, v in run_id_map.items()}
#
#     all_data = pd.DataFrame()
#
#     for gpkg_path in gpkg_file_paths:
#         file_data = load_and_prepare_data(gpkg_path)
#         all_data = pd.concat([all_data, file_data], ignore_index=True)
#         # sort it by run id
#
#     all_data['run_id'] = all_data['run_id'].map(run_id_map).astype(int)
#     all_data = all_data.sort_values(by='run_id')
#
#     # Define a color-blind friendly palette with a gradual color scheme
#     palette = sns.color_palette("Blues", n_colors=4)
#     hue_order = sorted(all_data['iou_thresh'].unique())
#
#     if scenario == 'a':
#         ax = sns.boxplot(x='run_id', y='ap50', hue='iou_thresh', data=all_data, notch=True, showmeans=True,
#                          meanprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red"},
#                          medianprops={"color": "white", "linestyle": "--"},
#                          palette=palette, hue_order=hue_order)
#
#         handles, labels = ax.get_legend_handles_labels()
#
#         # plt.title('Comparative Patch-level Average Precision (AP)')
#         plt.xlabel('Model ID')
#         plt.ylabel('AP')
#
#     elif scenario == 'b':
#         selectors = ['Patch resolution', 'scales', 'src_name']
#         unique_combinations = all_data[selectors].drop_duplicates()
#
#         num_plots = len(unique_combinations)
#         num_cols = 3
#         num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)
#
#         fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), constrained_layout=True)
#         axes = axes.flatten()
#
#         for i, (_, subset) in enumerate(unique_combinations.iterrows()):
#             subset_data = all_data[
#                 (all_data['Patch resolution'] == subset['Patch resolution']) &
#                 (all_data['scales'] == subset['scales']) &
#                 (all_data['src_name'] == subset['src_name'])
#                 ]
#
#             sns.boxplot(ax=axes[i], x='run_id', y='ap50', hue='iou_thresh', data=subset_data, notch=True,
#                         showmeans=True,
#                         meanprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red"},
#                         medianprops={"color": "white", "linestyle": "--", "linewidth": 2},
#                         palette=palette, hue_order=hue_order)
#
#             axes[i].set_title(f"{subset['Patch resolution']} | {subset['scales']} | {subset['src_name']}")
#             axes[i].set_xlabel('Run ID')
#             axes[i].set_ylabel('AP50')
#             if i != 0: axes[i].legend().set_visible(False)  # Show legend only on the first subplot
#
#         handles, labels = axes[0].get_legend_handles_labels()
#
#     elif scenario == 'c':
#         mean_data = all_data.groupby(['run_id', 'iou_thresh'])['ap50'].mean().reset_index()
#         ax = sns.barplot(x='run_id', y='ap50', hue='iou_thresh', data=mean_data, palette=palette, hue_order=hue_order)
#         ax.legend().set_visible(False)
#
#         # Apply tolerance to AP values before ranking
#         tolerance = 0.001
#         mean_data['ap50_rounded'] = mean_data['ap50'].apply(lambda x: np.round(x / tolerance) * tolerance)
#
#         # Calculate the rankings within each iou_threshold group using the rounded values
#         for iou in hue_order:
#             iou_group = mean_data[mean_data['iou_thresh'] == iou]
#             # Rank using rounded values and handle ties
#             rankings = iou_group['ap50_rounded'].rank(method='dense', ascending=False)
#
#             for index, row in iou_group.iterrows():
#                 rank = rankings[index]
#                 # Determine the marker based on rank
#                 if rank == 1:  # Highest
#                     marker, color, fontsize = "*", "red", 7
#                 elif rank == 2:  # Second highest
#                     marker, color, fontsize = r"$\blacktriangle$", "green", 4
#                 elif rank == 3:  # Third highest
#                     marker, color, fontsize = "+", "blue", 5
#                 else:
#                     continue  # No marker for ranks below third
#
#                 # Calculate position for the text annotation
#                 run_index = list(mean_data['run_id'].unique()).index(row['run_id'])
#                 hue_index = hue_order.index(iou)
#                 x_pos = run_index + (hue_index - (len(hue_order) - 1) / 2) * 0.2  # Adjusting offset for hue in grouped bar plot
#
#                 # Add text annotation on the bar, maker size 4
#                 ax.text(x_pos, row['ap50'] + 0.01, marker, ha='center', va='center', fontsize=fontsize, color=color)
#
#         handles, labels = ax.get_legend_handles_labels()
#
#         # plt.title('Mean Patch-level Average Precision (AP) by IoU Threshold with Tolerance')
#         plt.xlabel('Model ID')
#         plt.ylabel('Mean AP')
#
#     # Global legend for all subplots
#     plt.figlegend(handles, ['0.3', '0.5', '0.7', '0.9'], loc='lower center', ncol=len(hue_order),
#                   borderaxespad=0.1, borderpad=0.1, handletextpad=0.1, handlelength=1, columnspacing=0.5)
#     # reduce the margin between the legend and the subplots
#     plt.subplots_adjust(bottom=0.18)
#     plt.savefig(out_fp, dpi=300)


def extract_all_aps(in_fps, datasets=['test', 'val'], out_fp=None):
    df = []
    for i in in_fps:
        for j in datasets:
            df.append(extract_aps(i, j, None))
    df = pd.concat(df)
    df['regroup'] = df['region'] + '_' + df['res_scale']
    df.sort_values(by=['dataset_name', 'region', 'res_scale', 'iou_thresh'], inplace=True)
    df = df.pivot(index=['dataset_name', 'regroup'], columns='iou_thresh', values='ap')
    df.reset_index(inplace=True)
    # calculate the average for each row
    # df['mean'] = df[[0.3, 0.5, 0.7]].mean(axis=1)
    if out_fp:
        df.to_csv(out_fp, index=False)
    return df


def extract_aps(in_fp, dataset_name='test', out_fp=None):
    # extract ap metrics from log file for all in one models
    with open(in_fp) as f:
        lines = f.readlines()

        # start to search from the second place where 'INFO: Network:' appears, and search from the bottom
        for i in range(len(lines) - 1, 0, -1):
            if 'INFO: Not extracting images.' in lines[i]:
                break
        lines = lines[i + 2:]

        # Get iou threshold from the first line
        try:
            iou_thresh = lines[0].split('IOU = ')[-1].split('\n')[0]

            name = []
            ap = []
            region = []
            res_scale = []

            for l in lines:
                if 'INFO: ' in l:
                    line = l.split('INFO: ')[1]
                    if f'for {dataset_name} in' in line:
                        if 'AP' in line:
                            if f'AP50' in line:
                                line = line.split('AP50 for ')[-1]
                            elif f'AP' in line:
                                line = line.split('AP for ')[-1]
                            line = line.replace('scale:', 'scale')
                            region.append(line.split(dataset_name + ' in ')[-1][0:2])
                            name.append(line.split(': ')[0])
                            ap.append(line.split(': ')[-1].split('\n')[0])
                            res_scale.append(line.split(' at resolution ')[-1].split(': ')[0])

            df = pd.DataFrame({'iou_thresh': [iou_thresh] * len(ap),
                               'dataset_name': [dataset_name] * len(ap),
                               'region': region,
                               'res_scale': res_scale,
                               'ap': ap})

            # set the data types
            df['ap'] = df['ap'].astype(float)
            df['tp'] = df['ap'].astype(int)
            df['iou_thresh'] = df['iou_thresh'].astype(float)

            if out_fp is not None:
                if os.path.exists(out_fp):
                    df.to_csv(out_fp, mode='a', header=False, index=False)
                else:
                    df.to_csv(out_fp, index=False)
            return df
        except:
            pass


if __name__ == '__main__':
    viz_patch_ap50 = False
    metrics_extraction = True
    indTest_metrics_extraction = True
    ap50_extraction = False

    extract_ap_allinon = False

    import datetime

    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    if extract_ap_allinon:
        # extract all aps from logs.txt files for models trained with all datasets
        extract_all_aps(in_fps=[
            '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/evaluation_v20241022_data20240717_5c_iou0p7_autoResample_Oct23_18-22-00_7_maverickmiaow/logs.txt',
            '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/evaluation_v20241022_data20240717_5c_iou0p9_autoResample_Oct23_19-04-38_11_maverickmiaow/logs.txt',
            '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/evaluation_v20241022_data20240717_5c_iou0p3_autoResample_Oct23_17-38-25_13_maverickmiaow/logs.txt',
            '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS/evaluation_v20241022_data20240717_5c_iou0p5_autoResample_Oct23_16-39-49_5_maverickmiaow/logs.txt'],
                        out_fp='./tmp/allinone_aps.csv')

    # if viz_patch_ap50:
    #     wd = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/reports'
    #     # get folder name in path /mnt/raid5/DL_TreeHealth_Aerial/Merged/reports, with data20240801 in the name
    #     folder_list = [i for i in os.listdir(wd) if 'data20240801' in i and 'autoResample' not in i]
    #     gpkg_file_paths = [os.path.join(wd, i, j) for i in folder_list for j in os.listdir(os.path.join(wd, i)) if
    #                        'test' in j and j.endswith('.gpkg')]
    #     visualize_multiple_patch_ap(gpkg_file_paths, scenario='c', out_fp=f'./tmp/swiss_patch_ap50_barchart_{date_time}.png')
    #     visualize_multiple_patch_ap(gpkg_file_paths, scenario='a', out_fp=f'./tmp/swiss_patch_ap50_boxplot_{date_time}.png')

    if metrics_extraction:
        # Extract configs from logs folder
        wd = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS'
        run_metrics_fp = f'./tmp/swiss_eval_metrics_{date_time}.csv'
        batch_run(wd, prefix='run', out_fp=f'./tmp/configs_withCHexp_{date_time}.csv', evaluation_metrics=True)
        # Extract test metrics to a new csv
        # for swiss experiments
        extract_test_metrics(f'./tmp/configs_withCHexp_{date_time}.csv', run_metrics_fp, run_id_key='swiss-experiments')
        # for all runs
        extract_test_metrics(f'./tmp/configs_withCHexp_{date_time}.csv', f'./tmp/all_eval_metrics_{date_time}.csv', run_id_key=None)

        # # For regional models
        # wd = '/mnt/raid5/DL_TreeHealth_Aerial/Germany/logs_DeLfoRS'
        # batch_run(wd, prefix='patch', out_fp=f'./tmp/configs_DE_{date_time}.csv')

    if indTest_metrics_extraction:
        # Extract configs from logs folder for independent test runs
        wd = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS'
        batch_run(wd, path_filter=None, prefix='testAutoResample', out_fp=f'./tmp/configs_testAutoResample_{date_time}.csv', evaluation_metrics=True, ind_test=True)
        # Extract test metrics to a new csv
        df_ar = extract_test_metrics(f'./tmp/configs_testAutoResample_{date_time}.csv', f'./tmp/swiss_indTestAutoResample_metrics_{date_time}.csv')

        batch_run(wd, path_filter=None, prefix='testNoResample', out_fp=f'./tmp/configs_testNoResample_{date_time}.csv', evaluation_metrics=True, ind_test=True)
        # Extract test metrics to a new csv
        df_nr = extract_test_metrics(f'./tmp/configs_testNoResample_{date_time}.csv', f'./tmp/swiss_indTestNoResample_metrics_{date_time}.csv')

        # replacing metrics about count loss from run logs with the ones from ind test with auto resample
        metrics = pd.read_csv(run_metrics_fp)
        loss_cols = ['loss_count/test', 'loss_count_cumprod/test', 'loss_count_cumprod_0/test',
                     'loss_count_cumprod_1/test', 'loss_count_cumprod_10/test', 'loss_cumprod_comb/test',
                     'loss_wcount_comb/test', 'rMAE(%)', 'rMAE(%)_ORI']
        metrics = metrics.merge(df_ar, on=['run_id'], how='left', suffixes=('', '_indTest'))
        # fillna in columns with _indTest with the values from columns
        for i in loss_cols:
            metrics[f'{i}_indTest'] = metrics[f'{i}_indTest'].fillna(metrics[i])
        for i in loss_cols:
            metrics[i] = metrics[f'{i}_indTest']
        # drop columns with _indTest
        metrics = metrics.drop(columns=[i for i in metrics.columns if '_indTest' in i])
        # add metrics from ind test with no auto resample and rename as _ORI
        metrics = metrics.merge(df_nr, on=['run_id'], how='left', suffixes=('', '_ORI'))
        metrics.sort_values(by='iou/watershed_cumprod/test/1', ascending=False).to_csv(f'./tmp/swiss_eval_metrics_replaced_{date_time}.csv', index=False)

        # Get a subset of columns
        sub_cols = ['run_id', 'scenario', 'loss_count_cumprod/test', 'rMAE(%)', 'iou/watershed_cumprod/test/1',
                    'f1/watershed_cumprod/test/1', 'loss_count_cumprod/test_ORI', 'rMAE(%)_ORI_ORI',
                    'iou/watershed_cumprod/test/1_ORI', 'f1/watershed_cumprod/test/1_ORI']
        metrics[sub_cols].sort_values(by='iou/watershed_cumprod/test/1', ascending=False).to_csv(f'./tmp/swiss_eval_metrics_replaced_subset_{date_time}.csv', index=False)

        # Get a subset of columns and calculate mean and std
        metrics = metrics[sub_cols]
        # rename some ids
        rename_run_id = {'swiss-experiments_1_20240720': 'swiss-experiments_1old_20240720',
                         'swiss-experiments_2_20240721': 'swiss-experiments_2old_20240721',
                         'swiss-experiments_3_20240723': 'swiss-experiments_3old_20240723'
                         }
        metrics['run_id'] = metrics['run_id'].map(rename_run_id).fillna(metrics['run_id'])

        metrics['real_id'] = metrics['run_id'].apply(lambda x: x.split('_')[1])

        metrics['repeated_run'] = metrics['real_id'].apply(lambda x: x[-2:])
        metrics['repeated_run'] = metrics['repeated_run'].apply(lambda x: x if x in ['r2', 'r4', 'r1', 'r3', 'r5'] else 'r0')
        metrics['real_id'] = metrics.apply(lambda x: x['real_id'].replace(x['repeated_run'], '') if x['repeated_run'] in ['r2', 'r4', 'r1', 'r3', 'r5'] else x['real_id'], axis=1)

        # metrics = metrics[~metrics['run_id'].isin(['swiss-experiments_4r2_20240925', 'swiss-experiments_4r4_20240925'])]
        # calculate mean and std after group by real_id
        mean_ = metrics[metrics['repeated_run'].isin(['r0', 'r2', 'r4'])].groupby(['real_id']).agg('mean').reset_index()
        std_ = metrics[metrics['repeated_run'].isin(['r0', 'r2', 'r4'])].groupby(['real_id']).agg('std').reset_index()
        count_ = metrics[metrics['repeated_run'].isin(['r0', 'r2', 'r4'])].groupby(['real_id']).agg('count').reset_index()

        # drop nan in std
        mean_ = mean_[count_['run_id'] > 1].sort_values(by='iou/watershed_cumprod/test/1', ascending=False)
        std_ = std_[count_['run_id'] > 1].sort_values(by='iou/watershed_cumprod/test/1', ascending=False)
        count_ = count_[count_['run_id'] > 1].sort_values(by='iou/watershed_cumprod/test/1', ascending=False)

        # save mean and std to csv
        mean_.to_csv(f'./tmp/swiss_eval_metrics_replaced_mean_{date_time}.csv', index=False)
        std_.to_csv(f'./tmp/swiss_eval_metrics_replaced_std_{date_time}.csv', index=False)
        count_.to_csv(f'./tmp/swiss_eval_metrics_replaced_count_{date_time}.csv', index=False)


    if ap50_extraction:
        # Extract ap50 metrics from the evaluation logs
        out_fp = f'./tmp/swiss_ap50_{date_time}.csv'
        folder_list = [i for i in os.listdir('/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS') if
                       i.startswith('evaluation_swiss-experiments')]
        folder_list = sorted(folder_list)[::-1]
        df_list = []
        for fn in folder_list:
            df_list.append(extract_ap50(os.path.join('/mnt/raid5/DL_TreeHealth_Aerial/Merged/logs_DeLfoRS', fn, 'logs.txt'), None))
        df = pd.concat(df_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
        df.to_csv(out_fp, index=False)

        # Visualize the ap50 metrics
        # ap metrics for non-resampled test set
        run_id_list = [4, 906, 302, 9022, 9042, 202, 122, 21, 18, 5, 10, 0, 13, 6, 8, 16, 17, 102, 24, 907, 3]
        iou_thresh_list = [0.5]
        viz_ap50(out_fp, f'./tmp/swiss_ap50_{date_time}.png', dataset=['test'], name_ids=list(range(6, 12)), run_id_list=run_id_list, iou_thresh_list=iou_thresh_list)
        # ap metrics for resampled and non-resampled test set
        viz_ap50(out_fp, f'./tmp/swiss_ap50_autoResample_{date_time}.png', dataset=['test'], name_ids=list(range(36, 40)) + list(range(10, 12)), run_id_list=run_id_list, iou_thresh_list=iou_thresh_list)
        # # ap metrics by scales for resampled and non-resampled test set
        # viz_ap50(out_fp, f'./tmp/swiss_ap50_autoResample_{date_time}.png', dataset=['test'], name_ids=list(range(22, 32)))

        # # Append ap50 to metrics
        if indTest_metrics_extraction:
            metrics_fp = f'./tmp/swiss_eval_metrics_replaced_{date_time}.csv'
        else:
            metrics_fp = f'./tmp/swiss_eval_metrics_{date_time}.csv'
        metrics = pd.read_csv(metrics_fp)
        # replace baseline in run_id with 0
        metrics['run_id'] = metrics['run_id'].apply(lambda x: x.replace('baseline', '0'))

        rename_run_id = {'swiss-experiments_1_20240720': 'swiss-experiments_1old_20240720',
                         'swiss-experiments_2_20240721': 'swiss-experiments_2old_20240721',
                         'swiss-experiments_3_20240723': 'swiss-experiments_3old_20240723'
                         }
        metrics['run_id'] = metrics['run_id'].map(rename_run_id).fillna(metrics['run_id'])


        ap_fp = f'./tmp/swiss_ap50_{date_time}.csv'
        ap = pd.read_csv(ap_fp)
        ap['run_id'] = ap['run_id'] + '_' + ap['model_creation_date'].astype(str)
        ap['run_id'] = ap['run_id'].apply(lambda x: x.replace('evaluation_', ''))

        # Merge in AP for test dataset (non-resampled)
        # move iou_thresh to columns
        ap0 = ap[ap['name'] == 'test dataset (non-resampled)'].pivot(index='run_id', columns='iou_thresh', values='ap').reset_index()
        # rename columns to ap_iou
        ap0.columns = ['run_id'] + [f'AP{str(int(float(i) * 100))}_ORI' for i in ap0.columns[1:]]
        metrics = metrics.merge(ap0, on=['run_id'], how='left')

        # Merge in AP for resampled test dataset
        # move iou_thresh to columns
        ap1 = ap[ap['name'] == 'test dataset (resampled and non-resampled)'].pivot(index='run_id', columns='iou_thresh', values='ap').reset_index()
        # rename columns to ap_iou_resampled
        ap1.columns = ['run_id'] + [f'AP{str(int(float(i) * 100))}' for i in ap1.columns[1:]]
        metrics = metrics.merge(ap1, on=['run_id'], how='left')

        # remove evaluation from the keys in run_id_map
        run_id_map = {k.replace('evaluation_', ''): v for k, v in run_id_map.items()}
        # apply run_id_map to run_id
        metrics = metrics[metrics['run_id'].isin(run_id_map.keys())]
        metrics['run_id'] = metrics['run_id'].map(run_id_map).astype(int)
        metrics.to_csv(metrics_fp, index=False)

        # Get a subset of columns
        cols = ['run_id', 'scenario', 'loss_count_cumprod/test',  'rMAE(%)', 'iou/watershed_cumprod/test/1', 'f1/watershed_cumprod/test/1', 'AP30', 'AP50', 'AP70', 'AP90', 'loss_count_cumprod/test_ORI',  'rMAE(%)_ORI.1', 'iou/watershed_cumprod/test/1_ORI', 'f1/watershed_cumprod/test/1_ORI', 'AP30_ORI', 'AP50_ORI', 'AP70_ORI', 'AP90_ORI']
        metrics[cols].sort_values(by='iou/watershed_cumprod/test/1', ascending=False).to_csv(metrics_fp.split(date_time)[0] + 'subset_' + date_time + '.csv', index=False)
