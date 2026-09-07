from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        sep = list(text)
        sep.sort()
        unique = set()
        stoi = {}
        itos = {}
        i = 0
        for char in sep:
            if char in unique:
                continue
            
            stoi[char] = i
            itos[i] = char
            i += 1
            unique.add(char)
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        mapping = []
        for char in text:
            mapping += [stoi[char]]
        return mapping


    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        string = ""
        for mapping in ids:
            string += itos[mapping]
        return string
