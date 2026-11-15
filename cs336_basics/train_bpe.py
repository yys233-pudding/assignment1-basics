from collections import Counter, defaultdict
from itertools import pairwise
from os import PathLike

import regex as re


PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens)})
    assert vocab_size >= len(vocab), "Vocab size must accommodate the initial vocabulary"

    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    if special_tokens:
        # Match longer tokens first when a special token is a prefix of another.
        delimiters = sorted(special_tokens, key=len, reverse=True)
        pattern = "|".join(re.escape(token) for token in delimiters)
        chunks = re.split(pattern, data)
    else:
        chunks = [data]

    counter = Counter(match.group() for chunk in chunks for match in PAT.finditer(chunk))
    sequences = []
    for segment, count in counter.items():
        symbols = [vocab[value] for value in segment.encode("utf-8")]
        sequences.append((symbols, count))

    pair_counts = Counter()
    pair_to_indices = defaultdict(set)
    for idx, (symbols, count) in enumerate(sequences):
        for pair in pairwise(symbols):
            pair_counts[pair] += count
            pair_to_indices[pair].add(idx)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        max_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_symbol = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        vocab[len(vocab)] = new_symbol

        # Updating the index below can change the set for this pair.
        affected_indices = list(pair_to_indices[max_pair])
        for idx in affected_indices:
            symbols, count = sequences[idx]
            old_pairs = Counter(pairwise(symbols))

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == max_pair:
                    new_symbols.append(new_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            sequences[idx] = (new_symbols, count)

            new_pairs = Counter(pairwise(new_symbols))
            for pair, occurrences in old_pairs.items():
                remaining = new_pairs.get(pair, 0)
                pair_counts[pair] += (remaining - occurrences) * count
                if not remaining:
                    pair_to_indices[pair].remove(idx)
                    if not pair_to_indices[pair]:
                        del pair_to_indices[pair]
                        del pair_counts[pair]

            for pair, occurrences in new_pairs.items():
                if pair not in old_pairs:
                    pair_counts[pair] += occurrences * count
                    pair_to_indices[pair].add(idx)

    return vocab, merges
