# My GPT Built from Scratch

> A small character-level GPT built from scratch in PyTorch, based on concepts and implementations developed while completing the [NeetCode ML Course](https://neetcode.io/).
>
> **Built by Krish Vitavkar - September 2026**

This project started as a collection of solutions written while completing the NeetCode ML course. The course progressively introduces the components needed to build a GPT, from gradient descent and neural networks through embeddings, attention, transformers, and text generation.

I then extended the course implementations into a **fully runnable GPT project**, including a training pipeline, dataset, checkpointing, and interactive text generation.

The goal is not to reproduce a production-scale language model, but to understand and implement the core components of a GPT end-to-end.

---

## What it does

The model is a **character-level GPT** trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset.

Given a sequence of characters such as:

```text
ROMEO:
```

the model predicts the next character repeatedly to generate new text.

Example:

```text
Enter seed text (or type 'exit'/'quit' to stop): ROMEO:

Generated text:
ROMEO: What shall I do? I know not what to say...
```

The generator supports arbitrary seed text from the model's vocabulary and uses temperature-controlled multinomial sampling.

---

## Project Structure

```text
GPT-from-scratch/
│
├── model/                         # GPT architecture
│   ├── attention.py              # Self-attention head
│   ├── multi_head_attention.py   # Multi-headed self-attention
│   ├── transformer.py            # Transformer block
│   ├── gpt.py                    # GPT model
│   ├── normalization.py          # Layer normalization
│   ├── batch_normalization.py    # Batch normalization
│   ├── rms_normalization.py      # RMS normalization
│   ├── embeddings.py             # Embeddings
│   ├── positional_encoding.py    # Positional encoding
│   ├── kv_cache.py               # KV-cache implementation
│   └── grouped_query_attention.py# Grouped-query attention
│
├── data/                         # Data and NLP utilities
│   ├── tokenizer.py              # BPE tokenizer
│   ├── vocab.py                  # Character-level vocabulary
│   ├── loader.py                 # Batched data loader
│   ├── dataset.py                # GPT dataset preparation
│   ├── nlp_preprocessing.py      # NLP preprocessing
│   └── tokenizer_utils.py        # Tokenization utilities
│
├── foundations/                  # Neural network fundamentals
│   ├── neuron.py
│   ├── backprop.py
│   ├── mlp.py
│   ├── activations.py
│   ├── loss.py
│   ├── training_loop.py
│   ├── dead_relu_detector.py
│   └── ...
│
├── train.py                      # Train the GPT and save checkpoint
├── generate.py                   # Interactive text generation
├── checkpoint.pt                 # Pretrained model weights
├── requirements.txt
└── README.md
```

Some files in `model/`, `data/`, and `foundations/` are implementations produced during the NeetCode ML course. Not every implementation is necessarily used by the final Tiny Shakespeare GPT; several exist as standalone implementations of concepts explored during the course.

---

## Note: Course Code vs. Standalone Implementation

The original course exercises use fixed random seeds and deterministic behaviour to support automated grading. When converting these exercises into a functioning language model, repeated RNG resets interfered with parameter initialisation, dropout, and stochastic generation. These resets were removed from the standalone implementation so that training and sampling behave as expected.

The original implementation also rounded output logits before returning them. Since rounding has zero gradient almost everywhere, this prevented gradients from propagating correctly through the language-model head. The standalone implementation returns the raw logits.

---

# Quick Start

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Generate text immediately

A pretrained checkpoint is included with the repository, so you **do not need to train the model yourself**.

```bash
python generate.py
```

You will be prompted for a seed:

```text
Enter seed text (or type 'exit'/'quit' to stop): ROMEO:
```

The program will then generate text from the trained model.

You can continue entering different prompts without restarting the program:

```text
Enter seed text (or type 'exit'/'quit' to stop): ROMEO:

Generated text:
ROMEO: ...

Enter seed text (or type 'exit'/'quit' to stop): JULIET:

Generated text:
JULIET: ...

Enter seed text (or type 'exit'/'quit' to stop): exit
Exiting...
```

---

# Training from Scratch

The repository also contains the Tiny Shakespeare dataset and the complete training pipeline.

To train a new model:

```bash
python train.py
```

The training script:

1. Loads the Tiny Shakespeare dataset.
2. Builds a character-level vocabulary.
3. Converts the dataset into token IDs.
4. Constructs the GPT model.
5. Samples training batches.
6. Performs forward propagation.
7. Calculates cross-entropy loss.
8. Backpropagates gradients.
9. Updates the model using AdamW.
10. Saves the trained model and configuration to `checkpoint.pt`.

The checkpoint contains:

```text
model_state
stoi
itos
config
```

so that the generation script can reconstruct the model and vocabulary without needing to retrain.

---

# Model Architecture

The final model is a small decoder-style Transformer with:

```text
Vocabulary
    │
    ▼
Token Embedding ─────┐
                     ├──► Addition ──► Transformer Blocks
Position Embedding ──┘                         │
                                              ▼
                                         LayerNorm
                                              │
                                              ▼
                                         Linear LM Head
                                              │
                                              ▼
                                      Character logits
```

The model currently uses:

| Parameter | Value |
|---|---:|
| Context length | 64 |
| Model dimension | 128 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Optimizer | AdamW |
| Loss | Cross Entropy |
| Dropout | 0.2 |
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |

The model predicts one character at a time rather than using word or subword tokens.

---

# Training

Training uses next-token prediction.

For a sequence such as:

```text
ROMEO: Wh
```

the training targets are shifted by one character:

```text
Input:  ROMEO: Wh
Target: OMEO: Wha
```

The model therefore learns:

```text
P(next character | previous characters)
```

During generation, the predicted character is appended to the context and the process repeats autoregressively.

---

# Text Generation

Generation uses the following process:

```text
Seed text
    │
    ▼
Convert characters → token IDs
    │
    ▼
GPT
    │
    ▼
Take logits for final position
    │
    ▼
Apply temperature
    │
    ▼
Softmax
    │
    ▼
Sample next character
    │
    ▼
Append character to context
    │
    └──────────────► repeat
```

The generator uses **multinomial sampling** rather than always selecting the highest-probability character.

Temperature controls the randomness of generation:

| Temperature | Effect |
|---:|---|
| `0.5` | More deterministic, potentially repetitive |
| `0.8` | Balanced |
| `1.0` | Standard sampling |
| `1.2+` | More random and varied |

The default is:

```python
temperature=0.8
```

---

# From Course Solutions to a Working GPT

The original NeetCode ML course implementations are designed around automated testing. Many exercises use a structure similar to:

```python
class Solution:
    def some_method(...):
        ...
```

The submitted method is called directly by the course's test infrastructure.

That is useful for learning individual concepts, but it does not by itself create a standalone application.

I therefore adapted the implementations to create an actual end-to-end program.

### Changes made beyond the original course exercises

- Added a real training entry point to `train.py`.
- Added dataset loading and vocabulary construction.
- Added Tiny Shakespeare as training data.
- Added checkpoint saving.
- Added checkpoint loading to `generate.py`.
- Added an interactive generation loop.
- Added support for arbitrary seed text.
- Added temperature-based sampling.
- Added a pretrained checkpoint so generation can run without retraining.
- Removed deterministic RNG resets that were useful for automated course tests but interfered with normal model training and generation.
- Removed output-logit rounding that prevented gradients from flowing correctly through the model.
- Removed unnecessary repeated random-seed resets throughout the model implementation.
- Added the inference path needed to turn the course's model components into an actual text-generation program.

These changes mean the repository is intended to be both a **record of the course implementations** and a **working standalone GPT project**.

---

# Reproducibility vs. Real Training

One important difference between the original course environment and this project is the use of random seeds.

The course's automated tests require deterministic behaviour so that the same inputs produce the expected outputs.

A real training run, however, relies on randomness for things such as:

- parameter initialization,
- dropout,
- batch sampling,
- token sampling during generation.

Repeatedly resetting the random number generator during model execution can unintentionally produce identical initialization, dropout masks, or sampling behaviour.

For the standalone GPT, unnecessary repeated `torch.manual_seed()` calls were therefore removed so that the model can train and generate normally.

---

# Dataset

The model is trained on **Tiny Shakespeare**, a small corpus containing Shakespeare's works.

This makes it particularly useful for experimenting with a character-level language model because:

- the dataset is small enough to train locally;
- the vocabulary is small;
- the data contains dialogue, punctuation, capitalization, and multiple characters;
- the resulting model can produce recognisable Shakespeare-like text despite its small size.

The dataset is intentionally small. This project is primarily an educational implementation rather than an attempt to produce a competitive language model.

---

# Limitations

This is a **small educational GPT**, not a modern large language model.

The model has:

- a small parameter count;
- only a 64-character context window;
- character-level rather than subword tokenization;
- a relatively small training dataset;
- limited training compute.

As a result, generated text can be grammatically incorrect, repetitive, or nonsensical over longer sequences.

The model nevertheless demonstrates the core mechanism behind autoregressive Transformer language models:

```text
context → attention → representation → next-token probabilities
```

and repeats this process to generate a sequence.

---

# Course

This project was built while completing the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning).

The course progression covered:

### Mathematics

- Gradient descent
- Loss functions
- Activation functions
- Optimisation

### Neural Networks

- Neurons
- Backpropagation
- MLPs
- Training loops

### PyTorch

- Tensors
- Autograd
- Modules
- Optimisers

### NLP

- Tokenisation
- Embeddings
- Positional information
- Language modelling

### Transformers

- Self-attention
- Multi-head attention
- Transformer blocks
- Normalisation

### GPT

- Autoregressive prediction
- Next-token prediction
- Text generation

The final project combines these concepts into a working character-level GPT.

---

# Implemented Features

The current standalone GPT implementation includes:

* Character-level tokenisation
* Token embeddings
* Positional embeddings
* Causal self-attention
* Multi-head self-attention
* Transformer blocks
* Layer normalisation
* Feed-forward neural networks
* Dropout
* AdamW optimisation
* Cross-entropy next-character prediction
* Tiny Shakespeare training dataset
* Checkpoint saving and loading
* Autoregressive text generation
* Multinomial sampling
* Temperature-controlled generation
* Interactive seed prompts
* CPU-based inference and training

The model can therefore be trained from scratch and then used independently for text generation without needing to retrain.

---

# Additional Implementations

The repository also contains implementations of several techniques explored during the NeetCode ML course that are **not currently part of the final Tiny Shakespeare GPT**.

These include:

* BPE tokenisation
* KV-cache
* Grouped Query Attention (GQA)
* RMS Normalisation
* Batch Normalisation
* Additional data-loading and NLP utilities

These implementations are retained in the repository as part of the progression through the course and as potential building blocks for future experiments.

The final model intentionally uses a simpler character-level vocabulary and standard multi-head self-attention so that the core GPT architecture remains straightforward to understand.

---

# Future Improvements

Potential extensions to the current GPT include:

* [ ] Add validation loss and a train/validation split
* [ ] Add training-loss visualisation
* [ ] Add configurable command-line training arguments
* [ ] Add automatic device selection for CPU/GPU
* [ ] Add learning-rate scheduling
* [ ] Add top-k sampling
* [ ] Add top-p (nucleus) sampling
* [ ] Implement KV-cache during autoregressive generation
* [ ] Experiment with BPE/subword tokenisation
* [ ] Compare standard multi-head attention with Grouped Query Attention
* [ ] Experiment with larger datasets
* [ ] Increase model size and context length
* [ ] Add model parameter-count reporting
* [ ] Add benchmarking for training and inference speed
* [ ] Build a simple web interface for interacting with the model


# Why I Built This

The purpose of this project was to go beyond using a pre-built language model and understand what is actually happening inside a GPT.

Starting from basic neural-network components and progressively implementing:

```text
Gradient Descent
      ↓
Neural Networks
      ↓
Embeddings
      ↓
Attention
      ↓
Multi-Head Attention
      ↓
Transformers
      ↓
GPT
      ↓
Text Generation
```

provided a way to understand the architecture from the bottom up.

The final result is a small but complete language model that can be **trained from scratch, saved, loaded, and used interactively to generate text**.