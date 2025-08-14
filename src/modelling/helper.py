import json
import logging
import os
import random
import re
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from string import punctuation

import psutil
import torch
import torch.nn as nn
from torch import optim

from src.modelling.model_wrapper import ModelWrapper
from src.modelling.models import UNet
from src.modelling.models.map_prediction import UNetNorm


def initialize_rundir(base_run_dir, params, log_stdout=True):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    rs = random.Random(datetime.now())
    hash = rs.getrandbits(4)
    run_dir = Path(base_run_dir) / \
              f"{params}_{current_time}_{hash}_{socket.gethostname()}"
    try:
        os.makedirs(run_dir, exist_ok=True)
        logs_file = run_dir / "logs.txt"

        # BasicConfig must be called before any logs are written!
        logging.basicConfig(filename=logs_file, level=logging.INFO,
                            format='%(levelname)s: %(message)s')

        if log_stdout:
            root = logging.getLogger()
            sh = logging.StreamHandler(sys.stdout)
            root.addHandler(sh)

        logging.info(f"Writing logs to {logs_file}")
        return run_dir
    except Exception as e:
        print(e)
        return run_dir


def save_config_to_rundir(run_dir, config):
    conf = config._dump()
    config_file = run_dir / 'config.json'
    with open(config_file, 'w') as fp:
        json.dump(conf, fp, indent=4)
    logging.info(f"Config written to {config_file}")


def initialize_wrapped_model(config, model_params, device):
    if config.model_type == "unet_segmentation_models_pytorch":
        import segmentation_models_pytorch as smp

        if config.load:
            wrapped_model = ModelWrapper.load_model(
                smp.Unet, config.load, device)
            optimizer = optim.Adam(
                wrapped_model.model.parameters(), lr=config.lr, eps=config.opt_eps)
            optimizer = ModelWrapper.load_optimizer(
                config.load, optimizer, device)
            logging.info(f'Model loaded from {config.load}')
        else:
            wrapped_model = ModelWrapper(smp.Unet, **model_params)
            wrapped_model.model.to(device=device)
            optimizer = optim.Adam(
                wrapped_model.model.parameters(), lr=config.lr, eps=config.opt_eps)
            # logging.info('Network', wrapped_model.model)
    else:
        raise Exception(f'Unknowng model type: {config.model_type}')
    logging.info(f'''
        Device:          {device}
        Leaning rate:    {config.lr}
        f'\tall params: {sum(p.numel() for p in wrapped_model.model.parameters())}\n'
        f'\ttrainable params: {sum(p.numel() for p in wrapped_model.model.parameters() if p.requires_grad)}\n'
        ''')
    return wrapped_model, optimizer


def initialize_model(config, device):
    modelKLS = get_model(config)
    model, optimizer = modelKLS.load_from_config(config, device)

    return model, optimizer


def get_model(config):
    model_dict = {
        'unet': UNet,
        'unetnorm': UNetNorm,
    }
    if config.model_type.lower() in model_dict:
        return model_dict[config.model_type.lower()]
    else:
        raise Exception(f'Unknown model type: {config.model_type}')


def port_to_multiple_gpus(model, config):
    try:
        md = [int(i) for i in config.multi_devices]
        net_multi = nn.DataParallel(model, device_ids=md)
        model = net_multi
        logging.info(f'Using multiple GPU devices: {md}')
    except Exception:
        logging.error(
            f'Could not set multiple GPU devices ({config.multi_devices}). Using only single device ...')
    return model


def get_cpu_count(cpu):
    phy_cpu = psutil.cpu_count()  # logical=True)
    if cpu == -1 or cpu == 'all' or phy_cpu < cpu:
        cpu_count = phy_cpu
    else:
        cpu_count = cpu
    return cpu_count


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_params_as_str(config):
    # only log params different from config
    conf_dict_keys = config.__dict__
    params = f"{str(config.__class__.__name__)}"
    for key in conf_dict_keys:
        # ignore class internals and methods
        if key.startswith('__') or callable(getattr(config, key)):
            continue
        # ignore save related commands
        elif key in ["processed_dir", "save_patch_df", "save_samples", "load", "log_dir", "reset_head"]:
            continue
        # ignore device related commands
        elif key in ["num_workers", "multi_devices"]:
            continue

        default_value = getattr(config.__class__, key)
        value = getattr(config, key)
        if value != default_value:
            # remove weird characters
            value = re.sub(r'[' + re.escape(punctuation) + ']', " ", str(value))
            # remove multiple whitespaces and leading as well as trailing ones
            value = re.sub('\s+', ' ', value).strip()
            params += f"--{key}-{value}"

    return params.strip()


def save_wrapped_model_checkpoint(save_cp, wrapped_model, model_dir, optimizer, epoch, checkpoint_per_epoch):
    if save_cp:
        try:
            os.mkdir(model_dir)
            logging.info(f'Created checkpoint directory: {model_dir}')
        except OSError:
            pass
    if ((epoch + 1) % checkpoint_per_epoch) == 0:
        wrapped_model.save_model(wrapped_model.model, os.path.join(
            model_dir, f'CP_epoch{epoch + 1}.pth'), optm=optimizer)
        logging.info(f'Checkpoint {epoch + 1} saved !')


def save_checkpoint(save_cp, model, model_dir, optimizer, epoch, checkpoint_per_epoch):
    if save_cp:
        try:
            os.mkdir(model_dir)
            logging.info(f'Created checkpoint directory: {model_dir}')
        except OSError:
            pass
    if ((epoch + 1) % checkpoint_per_epoch) == 0:
        model.save_model(model, os.path.join(
            model_dir, f'CP_epoch{epoch + 1}.pth'), optm=optimizer)
        logging.info(f'Checkpoint {epoch + 1} saved !')


def clean_epoch_checkpoints(best_model_path, epoch, model_dir):
    # keep only the last 4 best-performing models
    try:
        shutil.move(best_model_path, os.path.join(
            model_dir, f'BestModel_{epoch:05d}.pth'))

        model_files = sorted([f for f in os.listdir(model_dir) if
                              os.path.isfile(os.path.join(model_dir, f)) and f.startswith('BestModel')])
        for filename in model_files[:-4]:
            filename_relPath = os.path.join(model_dir, filename)
            os.remove(filename_relPath)
    except Exception as e:
        pass


# https://stackoverflow.com/a/72898627/4095771
def set_diff_1d(t1, t2, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.

    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]
