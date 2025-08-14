import numpy as np
import torch
from config.base import BaseSegmentationConfig
from src.modelling import helper


class ModelsTestConfig1(BaseSegmentationConfig):
    model_type: str = "unet"
    backbone: str = "standard"
    initial_stride: int = 1
    norm_type_down = None
    norm_type_up = None
    loss_function: str = "ce"
    lr: float = 1e-4
    n_classes: int = 8
    in_channels: int = 3
    patch_size = 128
    sequential_stride = 128


class ModelsTestConfig1_b(BaseSegmentationConfig):
    model_type: str = "unet"
    backbone: str = "standard"
    initial_stride: int = 1
    norm_type_down = "bn"
    norm_type_up = "bn"
    loss_function: str = "ce"
    lr: float = 1e-4
    n_classes: int = 8
    in_channels: int = 3
    patch_size = 128
    sequential_stride = 128


class ModelsTestConfig2_b(BaseSegmentationConfig):
    model_type: str = "unetnorm"
    norm_type_in: str = "bn"
    norm_type_down = "bn"
    norm_type_up = "bn"
    backbone: str = "standard"
    initial_stride: int = 1
    loss_function: str = "ce"
    lr: float = 1e-4
    n_classes: int = 8
    in_channels: int = 3
    patch_size = 128
    sequential_stride = 128


class ModelsTestConfig3(BaseSegmentationConfig):
    model_type: str = "unet"
    backbone: str = "efficientnet_b2"
    initial_stride: int = 1
    loss_function: str = "ce"
    lr: float = 1e-4
    n_classes: int = 8
    in_channels: int = 3
    patch_size = 128
    sequential_stride = 128


def contains_norm(config):
    if config.norm_type_down is not None and config.norm_type_up is not None:
        return True
    if hasattr(config, 'norm_type_in'):
        if config.norm_type_in is not None:
            return True
    return False


def model_test(model, optm, config, device):
    cn = contains_norm(config)
    x = torch.tensor(np.random.random((2, 3, 256, 256)), dtype=torch.float32).to(device=device)
    model.train()
    y1 = model(x)
    model.eval()
    y2 = model(x)

    model.train()
    # Save using torch serialization
    torch.save(model, './model_serial.pth')
    model2 = torch.load('./model_serial.pth')
    y3 = model2(x)
    model2.eval()
    y4 = model2(x)
    model2.train()
    y5 = model(x)

    config.load = './test_model.pth'
    model.save_model(model, config.load, optm=optm)
    model3, _ = helper.initialize_model(config, device=device)
    model3.train()
    y6 = model3(x)
    model3.eval()
    y7 = model3(x)
    model3.train()
    y8 = model3(x)

    # import ipdb
    # ipdb.set_trace()
    assert torch.allclose(y1, y5)
    assert torch.allclose(y1, y3)
    if cn:
        assert not torch.allclose(y2,
                                  y4)  # They are not same since y2 has statistics from 1 batch while y4 has statistics from two batches
    else:
        assert torch.allclose(y2,
                              y4)  # Same since no norm
    assert torch.allclose(y1, y6)
    assert torch.allclose(y1, y8)
    assert torch.allclose(y4, y7)


def test_train_segmentation():
    device = helper.get_device()
    pass
    # for KLS in [ModelsTestConfig1, ModelsTestConfig1_b, ModelsTestConfig2_b, ModelsTestConfig3]:
    #     config = KLS()
    #     model, optm = helper.initialize_model(config, device)
    #     model_test(model, optm, config, device)


if __name__ == "__main__":
    test_train_segmentation()
