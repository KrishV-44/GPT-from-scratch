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

        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!


if __name__ == "__main__":
    # This block is what actually runs generation end-to-end.
    # The `Solution` class above only defines the sampling logic; nothing
    # in this file previously called it or loaded a model, which is why
    # `python generate.py` did nothing. It loads the checkpoint produced
    # by `python train.py`, so run that first.
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

    seed_text = "ROMEO:"
    seed_ids = [stoi[c] for c in seed_text if c in stoi] or [0]
    context = torch.tensor([seed_ids], dtype=torch.long)

    # Solution.generate() above resets its RNG to the same fixed state
    # after every sampled character (see the comment in that method). That's
    # there on purpose so the NeetCode grader gets reproducible output, but
    # it means the "randomness" never actually advances between steps -- once
    # the model settles into a confident state, the same draw gets reapplied
    # forever, producing loops like "tititititi...". This function is the
    # same sampling logic minus that reset, so randomness genuinely
    # progresses across the sequence. 'temperature' controls how random the
    # sampling is: <1.0 sharpens the distribution (safer, more repetitive),
    # >1.0 flattens it (more variety, more mistakes).
    def sample(model, new_chars, context, context_length, int_to_char, temperature=0.8):
        generated_text = ""
        with torch.no_grad():
            for _ in range(new_chars):
                context_cond = context[:, -context_length:]
                logits = model(context_cond)
                last_logits = logits[:, -1, :] / temperature
                probs = nn.functional.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token), dim=1)
                generated_text += int_to_char[next_token.item()]
        return generated_text

    generated = sample(
        model=model,
        new_chars=300,
        context=context,
        context_length=config["context_length"],
        int_to_char=itos,
        temperature=0.8,
    )

    print("Seed:", repr(seed_text))
    print("Generated text:")
    print(seed_text + generated)