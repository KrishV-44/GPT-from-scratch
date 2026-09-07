from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        return [self.tokenize(str(num), vocab) for num in numbers]


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.tokenize(text, vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        if not words:
            return 0.0
        
        total_tokens = self.count_tokens(text, vocab)
        return round(total_tokens / len(words), 4)
    

    def tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        n = len(text)
        max_len = max((len(k) for k in vocab), default=1)

        while i < n:
            matched = False
            # Try longest possible match first
            for length in range(min(max_len, n - i), 0, -1):
                sub = text[i : i + length]
                if sub in vocab:
                    tokens.append(sub)
                    i += length
                    matched = True
                    break

            # Fallback if character sequence isn't in vocab
            if not matched:
                tokens.append(text[i])
                i += 1

        return tokens
