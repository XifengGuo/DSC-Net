import numpy as np
import torch
import pickle
from main import ConvAE

db = 'yaleb'
if db == 'coil20':
    ae = ConvAE(channels=[1, 16], kernels=[3])  # coil20
elif db == 'coil100':
    ae = ConvAE(channels=[1, 50], kernels=[5])  # coil100
elif db == 'orl':
    ae = ConvAE(channels=[1, 3, 3, 5], kernels=[3, 3, 3])
elif db == 'yaleb':
    from yaleb import ConvAE
    ae = ConvAE(channels=[1, 10, 20, 30], kernels=[5, 3, 3])

state_dict = ae.state_dict()

weights = pickle.load(open('%s.pkl' % db, 'rb'), encoding='latin1')

for k1, k2 in zip(state_dict.keys(), weights.keys()):
    print(k1, k2)
    print(state_dict[k1].size(), weights[k2].shape)
    if weights[k2].ndim > 3:
        weights[k2] = np.transpose(weights[k2], [3, 2, 0, 1])
    state_dict[k1] = torch.tensor(weights[k2], dtype=torch.float32)
    print(state_dict[k1].size())

ae.load_state_dict(state_dict)
torch.save(state_dict, 'pretrained_weights_original/%s.pkl' % db)
print('Pretrained weights are converted and saved.')
