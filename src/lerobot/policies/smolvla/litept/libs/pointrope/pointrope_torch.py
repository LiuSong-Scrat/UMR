"""
PointROPE
Pure PyTorch implementation
"""

import torch

class PointROPE(torch.nn.Module):
    
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq 
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D,seq_len,device,dtype) not in self.cache:
            inv_freq = self.F0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()
            sin = freqs.sin()
            self.cache[D,seq_len,device,dtype] = (cos,sin)
        return self.cache[D,seq_len,device,dtype]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim==2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions, max_seqlen = None):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (xyz position of each token)
        output:
            * tokens after appplying PointROPE (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3)%3==0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3
        assert positions.ndim==3 and positions.shape[-1] == 3 # Batch, Seq, 3
        if max_seqlen == None:
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
        else: # use dynamic sequence length according to batched input
            cos, sin = self.get_cos_sin(D, max_seqlen+1, tokens.device, tokens.dtype)
        # split features into three parts along the feature dimension, and apply rope1d on each subspace
        x, y, z = tokens.chunk(3, dim=-1)
        x = self.apply_rope1d(x, positions[:,:,0], cos, sin)
        y = self.apply_rope1d(y, positions[:,:,1], cos, sin)
        z = self.apply_rope1d(z, positions[:,:,2], cos, sin)
        tokens = torch.cat((x, y, z), dim=-1)
        return tokens