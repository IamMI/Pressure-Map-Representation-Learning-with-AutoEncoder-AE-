import torch
import torch.nn as nn
import math
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, emb_size: int = 8*8, l: int = 300):
        super().__init__()

        self.positions = nn.Parameter(torch.randn(l , emb_size))
        
    def forward(self, x):
        return x+self.positions


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = 8*8, emb_size: int = 8*8, img_size: int = 160*120):
        self.patch_size = int(math.sqrt(patch_size))
        
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.positions = nn.Parameter(torch.randn(img_size//patch_size , emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x.unsqueeze(1))
        x += self.positions
        
        return x
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.feedforward = nn.Sequential(
                nn.Linear(emb_size, expansion * emb_size),
                nn.GELU(),
                nn.Dropout(drop_p),
                nn.Linear(expansion * emb_size, emb_size),      
        )

    def forward(self, x):
        return self.feedforward(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 8*8,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 indim: int = 8*8,
                ):
        super().__init__()

        self.layernorm = nn.LayerNorm(indim)
        self.multihead = nn.MultiheadAttention(emb_size, num_heads=4)
        self.dropout = nn.Dropout(drop_p)
        
        self.resnet2 = ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ),
        )
        self.key = nn.Linear(indim, emb_size)
        self.query = nn.Linear(indim, emb_size)
        self.value = nn.Linear(indim, emb_size)
        
        
    def forward(self, x):
        x = self.layernorm(x)
        
        # Calculate key query value
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        
        # Forward
        x,_ = self.multihead(key=key, query=query, value=value)
        x = self.dropout(x)
        x = self.resnet2(x)
        
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])