"""
Build Decoder for ViT
    input: (b, patch_num, 24)
    output: (b, patch_num, 8, 8) -> (b, 160, 120)
"""

import torch
import torch.nn as nn
import os
import sys

from transformer import TransformerEncoder, TokenEmbedding

def paste(x):
    """
    Concatenate an image
        input: (b, 300, 8, 8)
        output: (b, 160, 120)
    """
    b, _, _, _ = x.shape
    device = x.device
    output_height = 160
    output_width = 120
    block_size = 8

    big_block = torch.zeros(b, output_height, output_width).to(device=device)
    
    for i in range(300):
        row = (i // 15) * block_size
        col = (i % 15) * block_size
        big_block[:, row:row + block_size, col:col + block_size] = x[:, i, :, :]
    
    return big_block
    

class Decoder(nn.Module):
    def __init__(self, indim=24, depth=3, **kwargs):
        super().__init__()
        self.indim = indim
        
        self.fc = nn.Sequential(
            nn.Linear(indim, 8*8),
            nn.LayerNorm(8*8),
        )
        
        self.tokenembed = TokenEmbedding(emb_size=8*8, l=300)
        self.transformer = TransformerEncoder(depth=3, **kwargs)
        self.decoder = nn.Sequential(
            nn.Linear(8*8, 8*8),
            nn.Sigmoid(),
            nn.Linear(8*8, 8*8),
            nn.Sigmoid(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=1, out_channels=1, stride=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        b, l, _ = x.shape
        
        x = self.fc(x)
        x = self.tokenembed(x)
        x = self.transformer(x)
        x = self.decoder(x).reshape(b, l, 1, 8, 8)
        x = self.output(x).reshape(b, l, 8, 8)
        
        y = paste(x)
        return y
        
