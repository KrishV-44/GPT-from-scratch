import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None or self.cache_v is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
            
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if kv_cache is None:
            kv_cache = KVCache()
            
        prev_seq_len = kv_cache.cache_k.shape[1] if kv_cache.cache_k is not None else 0
        
        full_k, full_v = kv_cache.update(k, v)
        
        d_k = q.shape[-1]
        scores = q @ full_k.transpose(-2, -1) / (d_k ** 0.5)
        
        q_len = q.shape[1]
        total_len = full_k.shape[1]
        
        q_idx = torch.arange(prev_seq_len, total_len, device=x.device).unsqueeze(1)
        k_idx = torch.arange(total_len, device=x.device).unsqueeze(0)
        
        # Query position must be >= Key position
        causal_mask = q_idx < k_idx  # True for forbidden future positions
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = nn.functional.softmax(scores, dim=-1)
        output = attn_weights @ full_v
        
        return torch.round(output, decimals=4), kv_cache
