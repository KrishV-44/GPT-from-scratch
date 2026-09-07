import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        max_start_idx = len(data) - context_length
        
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            
            # Sample batch_size start indices
            start_indices = torch.randint(low=0, high=max_start_idx, size=(batch_size,))
            
            # Slice input (X) and target (Y) sequences
            x_batch = torch.stack([data[i : i + context_length] for i in start_indices])
            y_batch = torch.stack([data[i + 1 : i + 1 + context_length] for i in start_indices])
            
            # Forward pass
            logits = model(x_batch)  # Shape: (batch_size, context_length, vocab_size)
            
            # Reshape logits to (N, C) and targets to (N) for CrossEntropyLoss
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            
            # Backward pass & optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return round(loss.item(), 4)
