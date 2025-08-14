import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .backbones_conf import backbones
from .modules.standard import DoubleConvSkip
from .util import CustomPixelShuffle_ICNR, ConvNormActivation, get_act, get_norm_layer, icnr_init


class UNet(nn.Module):

    @torch.no_grad()
    def __init__(self,
                 n_classes,
                 in_channels,
                 backbone="efficientnet_b0",
                 patch_size=(480, 480),
                 initial_stride: int = "default",
                 norm_type_down: str = "bn",
                 norm_type_up: str = None,
                 upsampling: str = "pixelshuffle",
                 blur_final: bool = False,
                 blur: bool = False,
                 final_skip: bool = True,
                 activation="elu",
                 self_attention: bool = False):
        super().__init__()

        backbone_dict = backbones[backbone]

        if norm_type_down == "default":
            norm_type_down = backbone_dict["norm_type"]
        if norm_type_up == "backbone":
            norm_type_up = norm_type_down
        self.norm_type_down = norm_type_down
        self.norm_type_up = norm_type_up
        self.norm_layer_down = get_norm_layer(norm_type_down)
        self.norm_layer_up = get_norm_layer(norm_type_up)
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.blur = blur
        self.blur_final = blur_final
        self.self_attention = self_attention
        self.final_skip = final_skip
        if activation == "default":
            activation = backbone_dict["activation"]
        self.activation = activation
        self.activation_layer = get_act(activation)
        self.patch_size = patch_size
        if initial_stride == "default":
            initial_stride = backbone_dict["initial_stride"]
        self.initial_stride = initial_stride
        self.upsampling = upsampling
        self.backbone = backbone

        self.up_block = backbone_dict["up_block"]

        # first layer is stem, then blocks

        self.backbone_model = backbone_dict["model"](
            in_channels=self.in_channels, activation_layer=self.activation_layer,
            norm_layer=self.norm_layer_down, initial_stride=initial_stride
        )

        # for param in self.backbone_model.parameters():
        #     param.requires_grad = False
        self.backbone_model.eval()

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        # create dummy features to test network integrity
        img = torch.rand(2, in_channels, *patch_size).detach()

        # iterate through backbone_model to dynamically measure sizes
        x, outs = self.backbone_model(img)

        self.scale_factors = []
        ps = img.shape[2]  # assumes same reduction over height and width
        for out in outs:
            self.scale_factors.append(ps / out.shape[2])
            ps = out.shape[2]
        self.scale_factors.append(ps / x.shape[2])

        in_sizes = [out.size(1) for out in outs]

        # invert sizes and outputs
        cross_sizes = in_sizes[::-1]
        outs = outs[::-1]
        self.scale_factors = self.scale_factors[::-1]

        # init up scaling
        up_layers = []
        for i, cross_size in enumerate(cross_sizes):
            final = i == len(cross_sizes) - 1 and not self.final_skip
            up_in_c = x.size(1)
            do_blur = blur and (not final or blur_final)
            sa = i == 0 and self_attention
            unet_block = UnetBlockDeep(up_in_c, cross_size, self.activation_layer, norm_layer=self.norm_layer_up,
                                       blur=do_blur, self_attention=sa, upsampling=self.upsampling, final=final,
                                       scale_factor=self.scale_factors[i], block=self.up_block).eval()
            up_layers.append(unet_block)

            x = unet_block(x, outs[i])

        self.up_layers = nn.ModuleList(up_layers).eval()
        last_dim = x.size(1)

        if self.scale_factors[-1] != 1.0:
            last_dim, self.final_upsampling = get_upscaler(
                last_dim, self.scale_factors[-1], self.activation_layer, self.norm_layer_up, self.blur, self.upsampling
            )
            self.final_upsampling.eval()
            x = self.final_upsampling(x)
        else:
            self.final_upsampling = nn.Sequential()

        # init final output layers
        if self.final_skip:
            # add initial feature maps (not downsampled as last long skip connection)
            self.pre_final_conv = CatResBlock(
                last_dim + in_channels, last_dim + in_channels,
                self.norm_layer_up, self.activation_layer
            ).eval()
        else:
            self.pre_final_conv = PassBlock()
        x = self.pre_final_conv(x, img)

        self.last_conv = nn.Conv2d(x.size(1), self.n_classes, 1).eval()

        self.last_conv(x)

        logging.info(f'Network:\n'
                     f'\t{self.in_channels} inputs\n'
                     f'\t{self.n_classes} output channels (classes)\n'
                     f'\t{self.upsampling} network upsampling mode\n'
                     f'\t{self.activation} activation function\n'
                     f'\t{self.blur} Blur upscaling\n'
                     f'\t{self.blur_final} Blur final upscaling\n'
                     f'\t{self.norm_type_down} Norm down\n'
                     f'\t{self.norm_type_up if self.norm_type_down is not None else self.norm_type_down} Norm up\n'
                     f'\tall params: {sum(p.numel() for p in self.parameters())}\n'
                     f'\ttrainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n'
                     )

    def forward(self, img):
        x = img
        x, outs = self.backbone_model(x)

        outs = outs[::-1]

        for out, up_layer in zip(outs, self.up_layers):
            x = up_layer(x, out)
        x = self.final_upsampling(x)
        out = self.last_conv(self.pre_final_conv(x, img))
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
        self.freeze_(self.backbone_model.parameters(), not backbone)

        logging.info(f"other weights are frozen: {other}")
        self.freeze_(self.up_layers.parameters(), not other)
        self.freeze_(self.pre_final_conv.parameters(), not other)

        logging.info(f"head weights are frozen: {head}")
        self.freeze_(self.last_conv.parameters(), not head)

    @staticmethod
    def get_model_dict_from_config(st_dict):
        return {
            "n_classes": st_dict.get("n_classes"),
            "in_channels": st_dict.get("in_channels") if type(st_dict.get("in_channels")) == int else sum(
                st_dict.get("in_channels")),
            "backbone": st_dict.get("backbone"),
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
            del st_dict_init["patch_size"]
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

        st_dict['model_type'] = "unet"
        st_dict['n_classes'] = model.n_classes
        st_dict['in_channels'] = model.in_channels
        st_dict['upsampling'] = model.upsampling
        st_dict['activation'] = model.activation
        st_dict['norm_type_down'] = model.norm_type_down
        st_dict['norm_type_up'] = model.norm_type_up
        st_dict['self_attention'] = model.self_attention
        st_dict['patch_size'] = model.patch_size
        st_dict['blur'] = model.blur
        st_dict['blur_final'] = model.blur_final
        st_dict['backbone'] = model.backbone
        st_dict['final_skip'] = model.final_skip
        st_dict['initial_stride'] = model.initial_stride
        return st_dict

    @classmethod
    def save_model(cls, model, path, optm=None):
        st_dict = cls.get_save_dict(model, optm)
        torch.save(st_dict, path)


class UnetBlockDeep(nn.Module):
    "A quasi-UNet block, using upsampling."

    def __init__(self, up_in_c: int, x_in_c: int, activation_layer: nn.Module, norm_layer: nn.Module,
                 blur: bool = False, self_attention: bool = False, upsampling="pixelshuffle",
                 final: bool = False, scale_factor=2, block: nn.Module = DoubleConvSkip):
        super().__init__()
        up_in_shuf, upscaler = get_upscaler(up_in_c, scale_factor, activation_layer, norm_layer, blur, upsampling)
        self.upscaler = upscaler

        channel_in = up_in_shuf + x_in_c
        channel_out = channel_in if final else channel_in // 2
        self.block = block(
            channel_in, channel_out, activation_layer=activation_layer,
            norm_layer=norm_layer, self_attention=self_attention
        )
        self.relu = activation_layer

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.upscaler(up_in)
        ssh = skip.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, skip.shape[-2:], mode='nearest')

        cat_x = self.relu(torch.cat([up_out, skip], dim=1))

        return self.block(cat_x)


def get_upscaler(in_dim, scale_factor, activation_layer, norm_layer, blur: bool, upsampling: str):
    upscaler = []
    out_dim = in_dim
    for i in range(int(scale_factor // 2)):
        if upsampling == "pixelshuffle":
            upscaler.append(
                CustomPixelShuffle_ICNR(
                    out_dim, activation_layer, norm_layer, out_dim // 2, scale=2
                )
            )
        elif upsampling == "deconv":
            upscaler.append(nn.ConvTranspose2d(out_dim, out_dim // 2, 2, 2))
        elif upsampling == "nearest":
            upscaler.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling),
                ConvNormActivation(
                    out_dim, out_dim // 2, 1, activation_layer=activation_layer, norm_layer=norm_layer
                )
            ))
        else:
            upscaler.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode=upsampling, align_corners=True),
                ConvNormActivation(
                    out_dim, out_dim // 2, 1, activation_layer=activation_layer, norm_layer=norm_layer
                )
            ))
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        if blur:
            upscaler.extend([nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)])
        out_dim //= 2
    upscaler = nn.Sequential(*upscaler)
    return out_dim, upscaler


# Following the convention of classifier
# add input normalization to the classifier
class UNetNorm(UNet):
    def __init__(self, n_classes, in_channels,
                 backbone="efficientnet_b0", patch_size=(480, 480), initial_stride: int = "default",
                 norm_type_in: str = "bn", norm_type_down: str = "bn", norm_type_up: str = None,
                 upsampling: str = "pixelshuffle", blur_final: bool = False, blur: bool = False,
                 final_skip: bool = True, activation="elu", self_attention: bool = False, **kwargs):
        super().__init__(n_classes, in_channels, backbone, patch_size, initial_stride,
                         norm_type_down, norm_type_up, upsampling, blur_final, blur, final_skip,
                         activation, self_attention, **kwargs
                         )
        assert norm_type_in is not None
        self.norm_type_in = norm_type_in
        norm_layer_in = get_norm_layer(norm_type_in)

        self.norm_in = norm_layer_in(in_channels)

    @classmethod
    def get_save_dict(cls, model, optm=None):
        st_dict = super().get_save_dict(model, optm)
        st_dict["norm_type_in"] = model.norm_type_in
        st_dict['model_type'] = "unetnorm"
        return st_dict

    @classmethod
    def get_model_dict_from_config(cls, st_dict):
        model_dict = super().get_model_dict_from_config(st_dict)
        model_dict["norm_type_in"] = st_dict.get("norm_type_in")
        return model_dict

    def forward(self, input):
        input = self.norm_in(input)
        return super().forward(input)

    def reset_parameters(self):
        super().reset_parameters()
        self.norm_in.apply(self.backbone_model.init_weights)

    def freeze(self, backbone=True, other=True, head=False):
        super().freeze(backbone, other, head)
        self.freeze_(self.norm_in.parameters(), not other)


class ResBlock(nn.Module):
    def __init__(self, ni, nh, norm_layer, activation_layer):
        super().__init__()
        self.convs = nn.Sequential(
            ConvNormActivation(ni, nh, 3, norm_layer=norm_layer, activation_layer=activation_layer),
            ConvNormActivation(nh, ni, 3, norm_layer=norm_layer, activation_layer=activation_layer)
        )

    def forward(self, x): return self.convs(x) + x


class CatResBlock(ResBlock):
    def forward(self, x, prev_x):
        x = torch.cat([x, prev_x], 1)
        return self.convs(x) + x


class CatBlock(nn.Module):
    def __init__(self, block, **kwargs):
        self.block = block(**kwargs)

    def forward(self, x, prev_x):
        x = torch.cat([x, prev_x], 1)
        return self.block(x)


class PassBlock(nn.Module):
    def forward(self, x, *args):
        return x
