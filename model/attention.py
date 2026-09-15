import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        #torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        K = self.key_gen(embedded)
        Q = self.query_gen(embedded)
        V = self.value_gen(embedded)

        attention_dim = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (attention_dim**0.5)

        seq_len = embedded.shape[1]
        mask = torch.tril(
            torch.ones((seq_len, seq_len), device=embedded.device)
        )
        scores = scores.masked_fill(mask == 0, float("-inf"))

        probs = torch.softmax(scores, dim=2)

        output = torch.matmul(probs, V)
        return torch.round(output, decimals=4)
