import logging
import typing

import torch
import torch.nn as nn
from torch import optim

from .util import CustomPixelShuffle_ICNR, icnr_init

from segmentation_models_pytorch import Unet as smpUnet

import os


class UNetWithScalar(nn.Module):

    @torch.no_grad()
    def __init__(self,
                 n_classes,
                 in_channels,
                 backbone="resnet50",
                 scalar_counts=1, **kwargs):
        super().__init__()

        self.backbone_name = backbone
        backbone = smpUnet(backbone, encoder_weights='imagenet', in_channels=in_channels, classes=n_classes)
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        if kwargs.get("pretrained_ckpt") and os.path.exists(kwargs.get("pretrained_ckpt")):
            logging.info(f"Loading pretrained weights from {kwargs.get('pretrained_ckpt')}")
            encoder_state_dict = torch.load(os.path.join(kwargs.get("pretrained_ckpt"), "best_net_encoder.pth"))
            # check number of input channels
            if encoder_state_dict["conv1.weight"].shape[1] < in_channels:
                logging.info(f"Checkpoint does not have the same number of input channels, partial load of first conv.")
                new_bands = torch.zeros((encoder_state_dict["conv1.weight"].shape[0], in_channels - 3,
                                         encoder_state_dict["conv1.weight"].shape[2],
                                         encoder_state_dict["conv1.weight"].shape[3]))
                torch.nn.init.kaiming_normal_(new_bands)
                encoder_state_dict["conv1.weight"] = torch.cat([encoder_state_dict["conv1.weight"], new_bands], dim=1)

            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(torch.load(os.path.join(kwargs.get("pretrained_ckpt"), "best_net_decoder.pth")))
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.scalar_counts = scalar_counts

        scalar_features = 128
        self.scalar_ = nn.Sequential(
            nn.Conv2d(self.scalar_counts , scalar_features, 1),
            nn.BatchNorm2d(scalar_features),
            nn.ReLU(),
            nn.Conv2d(scalar_features, scalar_features, 1),
            nn.BatchNorm2d(scalar_features),
            nn.ReLU(),
        )

        # Additional scalar feature integration
        self.scalar_in = nn.Conv2d(scalar_features, in_channels, 1)

        bottle_feat = self.encoder.out_channels[-1]
        self.scalar_bottle = nn.Conv2d(scalar_features, bottle_feat, 1)

        dec_feat = self.decoder_channels[-1]
        self.scalar_out = nn.Conv2d(scalar_features, dec_feat, 1)

        self.last_conv = nn.Conv2d(16, self.n_classes, 1).eval()

        logging.info(f'Network:\n'
                     f'\t{self.in_channels} inputs\n'
                     f'\t{self.n_classes} output channels (classes)\n'
                     f'\tall params: {sum(p.numel() for p in self.parameters())}\n'
                     f'\ttrainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n'
                     )

    def forward(self, x, scalars: list):
        # using code from segmentation model https://github.com/qubvel/segmentation_models.pytorch/blob/3bf4d6ef2bc9
        # d41c2ab3436838aa22375dd0f23a/segmentation_models_pytorch/base/model.py#L24
        """Sequentially pass `x` trough model`s encoder, decoder and heads
        :param x: input tensor
        :param scalars: list of scalar values"""

        # Reshape the scalar input and process it through the scalar block
        self.scalar_counts = len(scalars)
        scalars = torch.stack(scalars, dim=1) if  self.scalar_counts > 1 else scalars[0]
        scalar = scalars.reshape(-1, self.scalar_counts, 1, 1).to(x.device)
        scalar = self.scalar_(scalar)
        # add scalar in
        x = self.scalar_in(scalar) + x

        features = self.encoder(x)
        # add scalar bottleneck
        features[-1] = self.scalar_bottle(scalar) + features[-1]

        decoder_output = self.decoder(*features)
        # add scalar out
        decoder_output = self.scalar_out(scalar) + decoder_output

        out = self.last_conv(decoder_output)

        return out

    def reset_parameters(self):
        def init_weights(m):
            self.backbone_model.init_weights(m)
            if isinstance(m, CustomPixelShuffle_ICNR):
                m[0][0].weight.data.copy_(icnr_init(m[0][0].weight.data))

        if self.backbone in ["swin-b", "swin-s"]:  # skip pretrained backbone_model
            self.up_layers.apply(init_weights)
            self.pre_final_conv.apply(init_weights)
            self.last_conv.apply(init_weights)
        else:  # reset everything
            self.apply(init_weights)

    def reset_head(self):
        self.last_conv.apply(self.backbone_model.init_weights)

    @staticmethod
    def freeze_(params, state):
        for param in params:
            param.requires_grad = state

    def freeze(self, backbone=True, other=True, head=False):
        logging.info(f"backbone weights are frozen: {backbone}")
        self.freeze_(self.encoder.parameters(), not backbone)

        logging.info(f"other weights are frozen: {other}")
        self.freeze_(self.decoder.parameters(), not other)

        logging.info(f"head weights are frozen: {head}")
        self.freeze_(self.last_conv.parameters(), not head)

    @staticmethod
    def get_model_dict_from_config(st_dict):
        return {
            "n_classes": st_dict.get("n_classes"),
            "in_channels": st_dict.get("in_channels") if type(st_dict.get("in_channels")) == int else sum(
                st_dict.get("in_channels")),
            "backbone": st_dict.get("backbone"),
            "pretrained_ckpt": st_dict.get("pretrained_ckpt"),
            "patch_size": st_dict.get("patch_size"),
            "initial_stride": st_dict.get("initial_stride"),
            "norm_type_down": st_dict.get("norm_type_down"),
            "norm_type_up": st_dict.get("norm_type_up"),
            "upsampling": st_dict.get("upsampling"),
            "blur_final": st_dict.get("blur_final"),
            "blur": st_dict.get("blur"),
            "final_skip": st_dict.get('final_skip'),
            "activation": st_dict.get("activation"),
            "self_attention": st_dict.get('self_attention'),
        }

    @classmethod
    def init_from_config(cls, st_dict):
        return cls(**cls.get_model_dict_from_config(st_dict))

    @classmethod
    def load_from_config(cls, config, device):
        """Initializes the model from config and the tries to load the weights in config.load."""
        model = cls.init_from_config(config)

        if config.load is not None:
            logging.info(f"Trying to load model from: {config.load}")
            st_dict_load = torch.load(config.load, map_location="cpu")
            st_dict_init = cls.get_save_dict(model)
            del st_dict_init["net_params"]
            # del st_dict_init["patch_size"]
            del st_dict_init["scalar_counts"]
            del st_dict_init['model_type']
            different_items = {
                k: st_dict_load[k] for k in st_dict_init if k in st_dict_load and st_dict_init[k] != st_dict_load[k]
            }

            strict = True

            if different_items.get("n_classes") or config.reset_head:
                logging.info(f"Loaded model has a different number of outputs, "
                             f"trying to load all weights except for last layer.")
                keys = [key for key in st_dict_load['net_params'].keys() if f"last_conv." in key]
                for key in keys:
                    del st_dict_load['net_params'][key]

                if different_items.get("n_classes"):
                    del different_items["n_classes"]
                strict = False
            assert len(different_items) == 0, (f"Loaded model cannot be loaded since it has a different config: " \
                                               f"{different_items}")

            model.load_state_dict(st_dict_load['net_params'], strict=strict)

        model.to(device=device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr, eps=config.opt_eps)
        if config.load is not None and st_dict_load.get("optm_params") is not None:
            if strict:  # must have changed last layer
                logging.info("Loading optimizer state.")
                optm_params = st_dict_load.get("optm_params")
                for i in range(len(optm_params["param_groups"])):
                    optm_params["param_groups"][i]["lr"] = config.lr
                    optm_params["param_groups"][i]["eps"] = config.opt_eps
                optimizer.load_state_dict(optm_params)
            else:
                logging.info("Didn't load optimizer since configs differ.")

        return model, optimizer

    @classmethod
    def load_from_path(cls, path, device):
        st_dict = torch.load(path, map_location=device)
        model = cls.init_from_config(st_dict)
        model.load_state_dict(st_dict['net_params'])
        model.to(device=device)
        return model

    @classmethod
    def get_save_dict(cls, model, optm=None):
        if isinstance(model, nn.DataParallel):
            model = model.module

        st_dict = {'net_params': model.state_dict()}

        if optm is not None:
            st_dict['optm_params'] = optm.state_dict()

        st_dict['model_type'] = "unet_with_scalar"
        st_dict['n_classes'] = model.n_classes
        st_dict['in_channels'] = model.in_channels
        st_dict['backbone'] = model.backbone_name
        st_dict['scalar_counts'] = model.scalar_counts
        return st_dict

    @classmethod
    def save_model(cls, model, path, optm=None):
        st_dict = cls.get_save_dict(model, optm)
        torch.save(st_dict, path)