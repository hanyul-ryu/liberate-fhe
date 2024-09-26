import os

import torch

if torch.cuda.is_available():
    from . import gpu
    print(">>>> Using GPU backend")
else:
    from . import cpu
    print(">>>> Using CPU backend")
