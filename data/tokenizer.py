from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # 2a. Count frequency of all adjacent token pairs
            pair_counts = defaultdict(int)
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            # 2b. Find the most frequent pair (break ties lexicographically)
            # Sort by frequency descending (-count), then pair ascending (p)
            best_pair = min(pair_counts.keys(), key=lambda p: (-pair_counts[p], p))

            # 2c. Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            merged_symbol = best_pair[0] + best_pair[1]

            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(merged_symbol)
                    i += 2  # Skip both merged tokens to avoid overlap
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

            # 2d. Record the merge as [token_a, token_b]
            merges.append([best_pair[0], best_pair[1]])

        # 3. Return the list of merges performed
        return merges
