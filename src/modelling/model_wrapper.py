import torch
from torch import nn as nn

class ModelWrapper():
    def __init__(
            self,
            nn_model_cls,
            **kwargs
    ):
        super(ModelWrapper, self).__init__()
        self.init_params = kwargs
        self.model = nn_model_cls(**kwargs)

    # def __getattr__(self, __name):
    #     if '__' not in __name and __name not in ModelWrapper.__dir__:
    # 	    return self.model.__name

    def freeze_encoder(self):
        for child in self.model.encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        return

    def unfreeze(self):
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = True
        return

    def save_model(self, model, path, optm=None):
        if isinstance(model, nn.DataParallel):
            model = model.module

        st_dict = {
            'init_params' :self.init_params,
            'model_params': model.state_dict()
            }

        if optm is not None:
            st_dict['optm_params'] = optm.state_dict()
        
        try:
            torch.save(st_dict, path)
            return 1

        except Exception as e:
            return 0
            
    @classmethod
    def load_model(cls, nn_model_cls, path, device):
        st_dict = torch.load(path, map_location=device)
        wrapped_model = ModelWrapper(nn_model_cls, **st_dict['init_params'])
        wrapped_model.model.load_state_dict(st_dict['model_params'])
        wrapped_model.model.to(device=device)
        return wrapped_model

    @classmethod
    def load_optimizer(cls, path, optm, device):
        st_dict = torch.load(path, map_location=device)
        optm.load_state_dict(st_dict['optm_params'])
        return optm