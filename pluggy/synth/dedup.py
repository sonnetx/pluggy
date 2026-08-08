"""
near-duplicate filter: word-ngram shingles + a small minhash signature as a
candidate index (docs sharing any minhash value get an exact jaccard check).
in-memory, single-threaded (only the pipeline's writer thread touches it),
fine for corpora up to low millions of docs. no llm calls here.
"""

import hashlib
import re


def _shingles(text: str, n: int) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {hash(tuple(words))}
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def _minhash(shingles: set[int], k: int) -> tuple[int, ...]:
    # k independent hashes via salted sha1 of the min shingle per salt.
    # cheap and deterministic; we don't need real permutation-quality minhash
    # for a candidate index that's backed by an exact jaccard check.
    sig = []
    for salt in range(k):
        best = min(
            int.from_bytes(
                hashlib.sha1(f"{salt}:{s}".encode()).digest()[:8], "big"
            )
            for s in shingles
        )
        sig.append(best)
    return tuple(sig)


class Deduper:
    def __init__(self, ngram: int = 13, threshold: float = 0.8, num_hashes: int = 8):
        self.ngram = ngram
        self.threshold = threshold
        self.num_hashes = num_hashes
        self.exact = set()          # sha1 of normalized text
        self.docs = []              # shingle sets, index = doc id
        self.buckets = {}           # minhash value -> [doc ids]

    def is_duplicate(self, text: str) -> bool:
        """check text against everything seen so far; record it if novel."""
        norm = " ".join(text.lower().split())
        digest = hashlib.sha1(norm.encode()).hexdigest()
        if digest in self.exact:
            return True
        shingles = _shingles(text, self.ngram)
        sig = _minhash(shingles, self.num_hashes)
        candidates = set()
        for v in sig:
            candidates.update(self.buckets.get(v, ()))
        for cand in candidates:
            other = self.docs[cand]
            inter = len(shingles & other)
            union = len(shingles) + len(other) - inter
            if union and inter / union >= self.threshold:
                return True
        # novel: record
        self.exact.add(digest)
        doc_id = len(self.docs)
        self.docs.append(shingles)
        for v in sig:
            self.buckets.setdefault(v, []).append(doc_id)
        return False
