import importlib

import timm
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl


def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    if mode in ['MPIIGaze', 'MPIIFaceGaze']:
        module = importlib.import_module(
            f'ptgaze.models.{mode.lower()}.{config.model.name}')
        model = module.Model(config)
    elif mode == 'ETH-XGaze':
        model = timm.create_model(config.model.name, num_classes=2)
        # model = Model()
        # print(config.model.name, model)
    else:
        raise ValueError
    device = torch.device(config.device)
    model.to(device)
    return model

def create_torch_model(config: DictConfig) -> torch.nn.Module:
    model = timm.create_model(config.MODEL.NAME,
                              pretrained=config.MODEL.PRETRAINED.USE_DEFAULT,
                              num_classes=2)
    if 'INIT' in config.MODEL:
        initialize_weight(config.MODEL.INIT, model)
    else:
        logger.warning('INIT key is missing in config.MODEL.')
    if 'PRETRAINED' in config.MODEL:
        load_weight(config.MODEL, model)
    else:
        logger.warning('PRETRAINED key is missing in config.MODEL.')
    return model

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = timm.create_model(config.MODEL.NAME,
        #                       pretrained=config.MODEL.PRETRAINED.USE_DEFAULT,
        #                       num_classes=2)

        # print('pytorch lightning model')

        self.model = timm.create_model('resnet18',
                              pretrained=True,
                              num_classes=2)

    def forward(self, x):
        return self.model(x)
