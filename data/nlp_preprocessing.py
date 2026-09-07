import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentences = positive + negative
        unique_words = sorted(set(word for sentence in all_sentences for word in sentence.split()))
        vocab = {word: idx + 1 for idx, word in enumerate(unique_words)}
        
        # 2. Encode each sentence by replacing words with their IDs
        encoded_sentences = []
        for sentence in all_sentences:
            encoded = [vocab[word] for word in sentence.split()]
            encoded_sentences.append(torch.tensor(encoded, dtype=torch.float32))
        
        # 3 & 4. Combine and pad sequences with 0s using batch_first=True
        padded_dataset = nn.utils.rnn.pad_sequence(encoded_sentences, batch_first=True, padding_value=0.0)
        
        return padded_dataset
