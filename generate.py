import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        generated_text = ""

        for i in range(new_chars):

            context_cond = context[:, -context_length:]
            
            logits = model(context_cond)
            last_logits = logits[:, -1, :]
            probs = nn.functional.softmax(last_logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            generator.set_state(initial_state)

            context = torch.cat((context, next_token), dim=1)
            
            token_id = next_token.item()
            generated_text += int_to_char[token_id]
        
        return generated_text


if __name__ == "__main__":
    import os
    from model.gpt import GPT

    HERE = os.path.dirname(os.path.abspath(__file__))
    CHECKPOINT_PATH = os.path.join(HERE, "checkpoint.pt")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}. Run `python train.py` first."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    config = checkpoint["config"]

    model = GPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        model_dim=config["model_dim"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Custom sampling without the RNG reset found in Solution.generate() 
    # (avoids infinite loops). Temperature controls variety: 
    # <1.0 = safer/repetitive, >1.0 = more varied/risky.
    def sample(model, new_chars, context, context_length, int_to_char, temperature=0.8):
        generated_text = ""
        with torch.no_grad():
            for j in range(new_chars):
                context_cond = context[:, -context_length:]
                logits = model(context_cond)
                last_logits = logits[:, -1, :] / temperature
                probs = nn.functional.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token), dim=1)
                generated_text += int_to_char[next_token.item()]
        return generated_text

    while True:
      seed_text = input("\nEnter seed text (or type 'exit'/'quit' to stop): ")

      if seed_text.strip().lower() in {"exit", "quit"}:
          print("Exiting...")
          break

      # Convert seed text into token IDs
      seed_ids = [stoi[c] for c in seed_text if c in stoi]

      # Fall back to token 0 if no characters are recognised
      if not seed_ids:
          print("No recognised characters in seed text.")
          continue
      
      unknown_chars = [c for c in seed_text if c not in stoi]
      if unknown_chars:
          print(f"Warning: characters not in vocabulary: {unknown_chars}")

      context = torch.tensor(
          [seed_ids],
          dtype=torch.long
      )

      generated = sample(
          model=model,
          new_chars=1000,
          context=context,
          context_length=config["context_length"],
          int_to_char=itos,
          temperature=0.8,
      )

      print("\nGenerated text:")
      print(seed_text + generated)