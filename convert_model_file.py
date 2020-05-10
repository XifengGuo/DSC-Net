import numpy as np
import torch
import pickle
from main import AE
ae = AE(1)
state_dict = ae.state_dict()

weights = pickle.load(open('coil20.pkl', 'rb'), encoding='latin1')

for k1, k2 in zip(state_dict.keys(), weights.keys()):
    print(k1, k2)
    if weights[k2].ndim > 3:
        weights[k2] = np.transpose(weights[k2], [3, 2, 0, 1])
    state_dict[k1] = torch.tensor(weights[k2], dtype=torch.float32)
    print(state_dict[k1].size())

ae.load_state_dict(state_dict)
torch.save(state_dict, 'pretrained_weights_original/coil20.pkl')