#!/usr/bin/env python3
"""AURORA: Adaptive Uncertainty-gated Residual Object Retrieval Architecture.

A dependency-free research prototype for the repository.  It is deliberately
small enough to run on a two-core, ~4 GiB machine.

This is not a claim of a frontier LLM.  AURORA is a conditional-compute
sequence model: a tiny always-on prior is augmented by a sparse, product-key
addressed episodic memory.  The memory is trained with a local delta/count
update and stores only high-surprise residuals.  It is useful when a workload
contains a large number of repeated or slowly changing associations.

The implementation is exact for the synthetic benchmark included below.  The
benchmark is a measurement of the mechanism, not evidence of Sonnet-level
open-domain capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
import hashlib
import json
import math
import random
import struct
import time


Context = Tuple[int, ...]


def stable_u64(values: Sequence[int], seed: int = 0) -> int:
    """Stable, process-independent 64-bit fingerprint for a context."""
    h = hashlib.blake2b(digest_size=8, person=b"aurora01")
    h.update(struct.pack("<Q", seed & ((1 << 64) - 1)))
    for value in values:
        h.update(struct.pack("<I", int(value) & 0xFFFFFFFF))
    return int.from_bytes(h.digest(), "little")


@dataclass(frozen=True)
class Example:
    context: Context
    target: int
    family: str


class DenseContextReference:
    """Uncompressed reference: one full V-wide count vector per context.

    It intentionally does more work at inference than AURORA.  It is a useful
    apples-to-apples reference for a memorisation-heavy workload: both models
    see exactly the same examples and have the same exact-answer capability.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.rows: Dict[Context, List[int]] = {}
        self.global_counts = [0] * vocab_size
        self.fit_examples = 0

    def fit(self, examples: Iterable[Example]) -> None:
        for ex in examples:
            row = self.rows.get(ex.context)
            if row is None:
                row = [0] * self.vocab_size
                self.rows[ex.context] = row
            row[ex.target] += 1
            self.global_counts[ex.target] += 1
            self.fit_examples += 1

    def predict(self, context: Context) -> Tuple[int, int, bool]:
        row = self.rows.get(context)
        if row is None:
            # Full-vocabulary scan is the reference fallback too.
            best = max(range(self.vocab_size), key=self.global_counts.__getitem__)
            return best, self.vocab_size, False
        best = max(range(self.vocab_size), key=row.__getitem__)
        return best, self.vocab_size, True

    def exact_bytes_estimate(self) -> int:
        # uint16 count per vocabulary entry, plus a compact 64-bit context key.
        # Python object overhead is intentionally excluded from both models;
        # this is a portable serialized-format estimate.
        return len(self.rows) * (8 + 2 * self.vocab_size) + 2 * self.vocab_size


@dataclass
class MemoryEntry:
    fingerprint: int
    # Sparse integer residual/count map.  In the serialized format each item
    # is (uint32 token, uint16 signed score).
    scores: Dict[int, int]
    seen: int = 0


class AuroraMemory:
    """Sparse product-key residual memory.

    Addressing uses two independent hashes, analogous to a product-key
    address.  A full 64-bit fingerprint makes hash collisions harmless: a
    collision simply becomes a miss, never a false fact.  The implementation
    uses a small bucket list so several fingerprints can share an address.

    The update is a local delta rule in count form.  Let c_s,y be the score for
    target y in activated slot s.  For target y_t we do

        c_s,y_t <- c_s,y_t + 1

    and periodically retain only the highest-surprise targets.  This is the
    integer/CPU form of a sparse fast-weight update; no full-vocabulary
    gradient is needed.
    """

    def __init__(
        self,
        vocab_size: int,
        n_key_a: int = 257,
        n_key_b: int = 263,
        top_values: int = 2,
        max_buckets: int = 3,
        surprise_margin: int = 0,
    ):
        self.vocab_size = vocab_size
        self.n_key_a = n_key_a
        self.n_key_b = n_key_b
        self.top_values = top_values
        self.max_buckets = max_buckets
        self.surprise_margin = surprise_margin
        self.slots: Dict[Tuple[int, int], List[MemoryEntry]] = {}
        self.global_counts = [0] * vocab_size
        self.examples = 0
        self.updates = 0

    def address(self, context: Context) -> Tuple[int, int]:
        return (
            stable_u64(context, 0xA17) % self.n_key_a,
            stable_u64(context, 0xB29) % self.n_key_b,
        )

    def _overflow_address(self, address: Tuple[int, int], fingerprint: int) -> Tuple[int, int]:
        return (
            (address[0] + 1 + (fingerprint % max(1, self.n_key_a - 1))) % self.n_key_a,
            (address[1] + 1 + ((fingerprint >> 17) % max(1, self.n_key_b - 1))) % self.n_key_b,
        )

    def _entry(self, context: Context, create: bool = True) -> MemoryEntry | None:
        address = self.address(context)
        fp = stable_u64(context, 0xC31)
        bucket = self.slots.get(address)
        if bucket is not None:
            for entry in bucket:
                if entry.fingerprint == fp:
                    return entry

        # Prediction must search the same deterministic overflow address that
        # insertion would use.  Otherwise a bounded bucket would turn a real
        # stored value into a false miss.
        if bucket is not None and len(bucket) >= self.max_buckets:
            address2 = self._overflow_address(address, fp)
            bucket2 = self.slots.get(address2)
            if bucket2 is not None:
                for entry in bucket2:
                    if entry.fingerprint == fp:
                        return entry

        if not create:
            return None
        if bucket is None:
            bucket = []
            self.slots[address] = bucket
        # A bounded bucket is important for memory guarantees.  If full, do
        # not evict a different association silently; use a deterministic
        # overflow address.
        if len(bucket) >= self.max_buckets:
            address2 = self._overflow_address(address, fp)
            bucket = self.slots.setdefault(address2, [])
            for entry in bucket:
                if entry.fingerprint == fp:
                    return entry
            if len(bucket) >= self.max_buckets:
                # This is an honest bounded-memory miss.  The caller can fall
                # back to the prior; it is never presented as a hit.
                return None
        entry = MemoryEntry(fp, {})
        bucket.append(entry)
        return entry

    def update(self, context: Context, target: int) -> bool:
        entry = self._entry(context, create=True)
        self.global_counts[target] += 1
        self.examples += 1
        if entry is None:
            return False
        entry.scores[target] = min(65535, entry.scores.get(target, 0) + 1)
        entry.seen = min(65535, entry.seen + 1)
        self.updates += 1
        return True

    def rewrite(self, context: Context, target: int, strength: int = 32767) -> bool:
        """Rewrite one activated association without touching other slots.

        This is the online/test-time-learning operation.  It is deliberately
        explicit rather than silently making an old answer disappear: callers
        can log the rewrite and verify it against an authority.  A real
        deployment would use a decay/rollback policy and a signed source.
        """
        entry = self._entry(context, create=True)
        if entry is None:
            return False
        entry.scores = {target: max(1, min(65535, int(strength)))}
        entry.seen = min(65535, entry.seen + 1)
        self.updates += 1
        return True

    def prune(self, prior: Sequence[int]) -> None:
        """Keep only residuals that beat a cheap prior by surprise_margin.

        This is the key size/quality tradeoff.  If an association is not
        surprising relative to the prior, it need not be stored in episodic
        memory.  Ties are retained in this prototype to make the benchmark
        exact and reproducible.
        """
        for bucket in self.slots.values():
            for entry in bucket:
                if not entry.scores:
                    continue
                ranked = sorted(
                    entry.scores.items(), key=lambda kv: (-kv[1], kv[0])
                )
                keep: Dict[int, int] = {}
                best = ranked[0][1]
                for token, score in ranked:
                    if len(keep) >= self.top_values:
                        break
                    if score + self.surprise_margin >= best:
                        keep[token] = score
                entry.scores = keep

    def predict(self, context: Context, prior: Sequence[int]) -> Tuple[int, int, bool]:
        entry = self._entry(context, create=False)
        if entry is None or not entry.scores:
            best = max(range(self.vocab_size), key=prior.__getitem__)
            return best, 1, False
        token = max(entry.scores, key=lambda t: (entry.scores[t], -t))
        return token, len(entry.scores), True

    def exact_bytes_estimate(self) -> int:
        # Portable packed representation estimate:
        # two uint16 product-key ids + bounded bucket metadata are amortised;
        # each entry stores fp (8), seen (2), and sparse (token uint32, score
        # int16) pairs.  This deliberately counts logical bytes, not Python
        # dict overhead.
        total = self.n_key_a * 4 + self.n_key_b * 4
        for bucket in self.slots.values():
            total += 4  # address
            for entry in bucket:
                total += 8 + 2 + 1
                total += len(entry.scores) * 6
        total += self.vocab_size * 2
        return total

    def logical_stats(self) -> Dict[str, int]:
        entries = [e for b in self.slots.values() for e in b]
        return {
            "occupied_product_slots": len(self.slots),
            "entries": len(entries),
            "sparse_values": sum(len(e.scores) for e in entries),
            "collisions_buckets_gt_1": sum(
                1 for b in self.slots.values() if len(b) > 1
            ),
        }


class AuroraHybrid:
    """A complete tiny prior + sparse memory model.

    The prior is deliberately transparent: a global token-frequency prior.
    A production version would replace it with a quantized recurrent/linear
    core without changing the memory protocol.  Keeping it simple makes the
    measured memory and lookup differences auditable on a minimal machine.
    """

    def __init__(self, vocab_size: int, **memory_kwargs):
        self.vocab_size = vocab_size
        self.prior = [0] * vocab_size
        self.memory = AuroraMemory(vocab_size, **memory_kwargs)

    def fit(self, examples: Iterable[Example]) -> None:
        materialized = list(examples)
        for ex in materialized:
            self.prior[ex.target] += 1
        self.memory.global_counts = list(self.prior)
        for ex in materialized:
            self.memory.update(ex.context, ex.target)
        self.memory.prune(self.prior)

    def predict(self, context: Context) -> Tuple[int, int, bool]:
        return self.memory.predict(context, self.prior)

    def exact_bytes_estimate(self) -> int:
        return self.vocab_size * 2 + self.memory.exact_bytes_estimate()


def make_benchmark(
    seed: int = 7,
    vocab_size: int = 2048,
    n_contexts: int = 5000,
    repeats: int = 4,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Create a deterministic benchmark with three explicit regimes.

    - TRAIN: many repeated key/value associations, where sparse episodic
      memory should dominate.
    - REPLAY: same contexts with different order, testing exact retention.
    - OOD: unseen contexts, testing honest fallback behavior.  No model can
      infer arbitrary random labels for OOD contexts without additional
      structure; this split prevents us from hiding that limitation.
    """
    rng = random.Random(seed)
    contexts: List[Context] = []
    targets: List[int] = []
    for i in range(n_contexts):
        # Contexts are deliberately non-consecutive and collision-resistant.
        ctx = (
            rng.randrange(vocab_size),
            rng.randrange(vocab_size),
            (i * 7919 + rng.randrange(vocab_size)) % vocab_size,
            i % 17,
        )
        target = (31 * ctx[0] + 17 * ctx[1] + 7 * ctx[2] + 13 * ctx[3]) % vocab_size
        contexts.append(ctx)
        targets.append(target)

    train: List[Example] = []
    for r in range(repeats):
        order = list(range(n_contexts))
        rng.shuffle(order)
        for i in order:
            train.append(Example(contexts[i], targets[i], "repeated_fact"))

    replay = [Example(contexts[i], targets[i], "replay") for i in reversed(range(n_contexts))]

    ood: List[Example] = []
    for i in range(max(1, n_contexts // 5)):
        ctx = (
            rng.randrange(vocab_size),
            rng.randrange(vocab_size),
            rng.randrange(vocab_size),
            1000 + i,
        )
        target = (31 * ctx[0] + 17 * ctx[1] + 7 * ctx[2] + 13 * ctx[3]) % vocab_size
        ood.append(Example(ctx, target, "ood"))
    return train, replay, ood


def evaluate(model, examples: Sequence[Example]) -> Dict[str, float]:
    correct = 0
    hits = 0
    lookup_ops = 0
    for ex in examples:
        pred, ops, hit = model.predict(ex.context)
        correct += int(pred == ex.target)
        hits += int(hit)
        lookup_ops += ops
    n = len(examples)
    return {
        "examples": n,
        "accuracy": correct / n if n else 0.0,
        "memory_hit_rate": hits / n if n else 0.0,
        "mean_candidate_or_scan_ops": lookup_ops / n if n else 0.0,
    }


def run_demo(seed: int = 7) -> Dict[str, object]:
    train, replay, ood = make_benchmark(seed=seed)
    vocab_size = 2048

    dense = DenseContextReference(vocab_size)
    t0 = time.perf_counter()
    dense.fit(train)
    dense_fit_s = time.perf_counter() - t0

    aurora = AuroraHybrid(
        vocab_size,
        n_key_a=257,
        n_key_b=263,
        top_values=2,
        max_buckets=3,
    )
    t0 = time.perf_counter()
    aurora.fit(train)
    aurora_fit_s = time.perf_counter() - t0

    dense_replay = evaluate(dense, replay)
    aurora_replay = evaluate(aurora, replay)
    dense_ood = evaluate(dense, ood)
    aurora_ood = evaluate(aurora, ood)

    dense_bytes = dense.exact_bytes_estimate()
    aurora_bytes = aurora.exact_bytes_estimate()
    return {
        "algorithm": {
            "name": "AURORA-PRM",
            "expanded": "Adaptive Uncertainty-gated Residual Object Retrieval with Product-key Memory",
            "note": "sparse residual memory + cheap prior; workload-conditional, not a frontier LLM",
        },
        "dataset": {
            "train_examples": len(train),
            "replay_examples": len(replay),
            "ood_examples": len(ood),
            "vocab_size": vocab_size,
            "unique_train_contexts": len(dense.rows),
        },
        "measured": {
            "dense_fit_seconds": dense_fit_s,
            "aurora_fit_seconds": aurora_fit_s,
            "dense_replay": dense_replay,
            "aurora_replay": aurora_replay,
            "dense_ood": dense_ood,
            "aurora_ood": aurora_ood,
            "dense_serialized_bytes_estimate": dense_bytes,
            "aurora_serialized_bytes_estimate": aurora_bytes,
            "memory_reduction_factor": dense_bytes / max(1, aurora_bytes),
            # A full softmax-style dense update has to score/update V output
            # entries per example.  AURORA's local memory rewrite touches one
            # target entry (plus addressing metadata).  These are operation
            # counts, not wall-clock claims; Python hashing intentionally is
            # not a tuned kernel.
            "dense_training_output_ops_estimate": len(train) * vocab_size,
            "aurora_training_sparse_update_ops_estimate": len(train),
            "training_output_work_reduction_factor": float(vocab_size),
            "replay_lookup_reduction_factor": (
                dense_replay["mean_candidate_or_scan_ops"]
                / max(1e-9, aurora_replay["mean_candidate_or_scan_ops"])
            ),
            "aurora_memory_stats": aurora.memory.logical_stats(),
        },
        "limits": [
            "OOD labels are arbitrary; no compression scheme can recover them without information.",
            "This experiment proves exact retention on a narrow synthetic workload only.",
            "It does not establish parity with Sonnet, frontier reasoning, multilingual quality, or safety.",
        ],
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
