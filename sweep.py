#!/usr/bin/env python3
"""Small deterministic scaling table for the AURORA-PRM prototype."""

from aurora import AuroraHybrid, DenseContextReference, make_benchmark


def main() -> None:
    train, replay, _ = make_benchmark(vocab_size=2048, n_contexts=1000, repeats=2)
    print("V\tdense_bytes\taurora_bytes\tmemory_x\tdense_scan\taurora_candidates")
    for vocab in (32, 128, 512, 2048, 8192, 16384):
        # The contexts/targets above are in the 2048-token domain.  This sweep
        # only measures the representation/lookup scaling, so clamp targets
        # into the tested vocabulary and keep contexts unchanged.
        data = [
            type(ex)(ex.context, ex.target % vocab, ex.family)
            for ex in train
        ]
        dense = DenseContextReference(vocab)
        dense.fit(data)
        aurora = AuroraHybrid(vocab, n_key_a=257, n_key_b=263)
        aurora.fit(data)
        dense_b = dense.exact_bytes_estimate()
        aurora_b = aurora.exact_bytes_estimate()
        print(
            f"{vocab}\t{dense_b}\t{aurora_b}\t"
            f"{dense_b / max(1, aurora_b):.1f}\t{vocab}\t1"
        )


if __name__ == "__main__":
    main()
