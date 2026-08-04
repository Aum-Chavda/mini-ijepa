from src.models.base import BaseEncoder, BasePredictor
print('BaseEncoder and BasePredictor imported OK')

import torch.nn as nn

class BadEncoder(BaseEncoder):
    pass  # forgot to implement forward and embed_dim

try:
    e = BadEncoder()
except TypeError as ex:
    print('ABC caught missing method:', ex)