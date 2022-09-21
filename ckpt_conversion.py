import torch

ckpt = torch.load('models/epoch=0009.ckpt', map_location='cpu')
state_dict = ckpt['state_dict']
for key in list(state_dict.keys()):
    state_dict[key[6:]] = state_dict[key]
    del state_dict[key]
torch.save({'model': state_dict}, 'models/ir_trained_220825.pth')