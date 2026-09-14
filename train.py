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


if __name__ == "__main__":
    # This block is what actually runs the training end-to-end.
    import os
    from data.vocab import Solution as VocabSolution
    from model.gpt import GPT

    HERE = os.path.dirname(os.path.abspath(__file__))
    CORPUS_PATH = os.path.join(HERE, "data", "tinyshakespeare.txt")
    CHECKPOINT_PATH = os.path.join(HERE, "checkpoint.pt")

    with open(CORPUS_PATH, "r") as f:
        text = f.read()

    vocab_solution = VocabSolution()
    stoi, itos = vocab_solution.build_vocab(text)
    encoded = vocab_solution.encode(text, stoi)
    data_tensor = torch.tensor(encoded, dtype=torch.long)

    config = {
        "vocab_size": len(stoi),
        "context_length": 64,
        "model_dim": 128,
        "num_blocks": 4,
        "num_heads": 4,
    }

    model = GPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        model_dim=config["model_dim"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
    )

    trainer = Solution()
    final_loss = trainer.train(
        model=model,
        data=data_tensor,
        epochs=8000,
        context_length=config["context_length"],
        batch_size=32,
        lr=3e-4,
    )

    print(f"Training complete. Final loss: {final_loss}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": config,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
