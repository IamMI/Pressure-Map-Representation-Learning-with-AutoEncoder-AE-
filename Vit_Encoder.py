"""
Import from https://github.com/FrancescoSaverioZuppichini/ViT/blob/main/README.ipynb
Build ViT Encoder
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from transformer import TransformerEncoder, PatchEmbedding
    

class Squeeze(nn.Sequential):
    def __init__(self, emb_size, outdim):
        super().__init__(
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, outdim),
        )
        
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 1,
                patch_size: int = 8*8,
                emb_size: int = 8*8,
                img_size: int = 160*120,
                depth: int = 12,
                outdim: int = 24,
                ):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size),
            Squeeze(emb_size, outdim),
        )

