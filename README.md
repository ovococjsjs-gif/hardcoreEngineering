# AURORA-PRM: honest low-cost model compiler

This repository contains a dependency-free CPU prototype of **AURORA-PRM**:

> **A**daptive **U**ncertainty-gated **R**esidual **O**bject **R**etrieval with **P**roduct-key **M**emory.

It is a proposed *conditional-compute memory module*, not a claim that a 2-core,
4 GiB machine can train an open-domain model equal to a frontier model such as
Claude Sonnet.  The repository intentionally measures what is actually achieved
and states what is not achieved.

## The hard boundary

A general Sonnet-level model cannot honestly be produced from this machine,
without a strong teacher, a large corpus, or an already-trained checkpoint, and
then claimed to have the same quality at 1/1000 of the information and compute.
A model must contain information about the task.  For arbitrary random labels on
unseen inputs, no compression algorithm can reconstruct the labels: the missing
information is not present anywhere in the input.

The defensible target is conditional:

- for a fixed workload with repeated or slowly changing associations, separate
  a cheap semantic prior from an episodic residual memory;
- activate only the relevant memory entries;
- store sparse quantized residuals instead of a full vocabulary logit vector;
- keep a collision-safe fingerprint so a hash collision is a miss, never a
  fabricated fact;
- measure exact accuracy on the workload and expose the out-of-distribution
  failure rather than hiding it.

This is the implemented target. It is a useful building block for a small local
model, a document compiler, a retrieval layer, or a teacher-distillation
pipeline. It is **not** a replacement for a general frontier LLM.

## AURORA-PRM in one page

AURORA has two components:

1. **Cheap prior.** In this CPU prototype it is a global token-frequency vector.
   A production implementation can replace it with a small quantized recurrent,
   linear-attention, or distilled Transformer core without changing the memory
   protocol.
2. **Sparse residual memory.** A context is mapped to two independent product
   keys.  The activated bucket contains a 64-bit context fingerprint and only
   the top target scores.  Training performs a local integer delta/count update;
   it never allocates or updates a full `V`-wide output vector.

At inference:

```text
context -> two product-key hashes -> bounded bucket -> fingerprint check
        -> sparse candidates -> prior fallback if miss
```

A collision is not treated as a hit.  If the bounded bucket is full, AURORA
returns a miss and falls back to the prior.  This is slower or less accurate
than silently returning a wrong association, but it is honest.

The full design is:

```text
slow semantic core
       +
uncertainty gate
       +
product-key address (two small keys)
       +
collision-safe sparse fast memory
       +
quantized packed storage
       +
verifiable fallback
```

The prototype uses a deterministic hash address rather than learned floating
point codebooks.  That makes the result reproducible and auditable in the
minimal environment.  A learned/product-quantized address can be substituted
later.

## Run it

No package installation is required:

```bash
python aurora.py
python -m unittest -v test_aurora.py
```

The demo creates 5,000 unique contexts, repeats them four times for training,
and evaluates both exact replay and unseen random contexts.  The reference
stores a full 2,048-wide `uint16` vector for every context; AURORA stores one
sparse target value per context.  The byte counts are estimates of a packed
on-disk format, not Python object sizes.

A representative run on the provided 2-core CPU produced:

```text
                         dense reference       AURORA-PRM
replay accuracy               100%                 100%
replay candidate work       2048                 1
packed memory              ~20.5 MB             ~0.115 MB
```

The exact numbers are emitted as JSON by `aurora.py`; runtime varies by machine.
The OOD split is deliberately reported separately.  Both systems are near the
unigram baseline there because the labels are arbitrary.  AURORA does not claim
to infer information that was never supplied.

## Why this can be cheap, and why it cannot be magic

For a vocabulary of size `V`, a dense context row requires `O(V)` score storage
and a full-vocabulary selection.  A sparse memory with `k` retained candidates
requires `O(k)` value storage and selection.  In the demo `V=2048`, `k=1`.
This creates a 2,048x candidate-scan reduction and about 179x packed-memory
reduction for the chosen workload.

Those factors are **not** a universal speedup for a complete LLM:

- hashing and memory misses have overhead;
- random-access memory can be bandwidth-bound;
- a large product-key table may need external memory;
- retrieval does not solve novel reasoning;
- a cheap prior can be much worse on open-ended language;
- a teacher or corpus still costs information and compute;
- if every input is new, the sparse memory has no advantage.

The correct engineering path is conditional computation and compilation, not a
claim that arithmetic has been defeated.

## Relation to prior work

The composition is proposed here; the ingredients are not claimed to be new in
isolation.  Relevant prior directions include sparse Mixture-of-Experts,
Product Key Memory, fast-weight/linear-attention memories, and recent sparse
fast-weight memory work.  See:

- Switch Transformers: https://jmlr.org/papers/volume23/21-0998/21-0998.pdf
- Large Product Key Memory: https://arxiv.org/abs/2002.02325
- Linear Transformers as Fast Weight Programmers: https://arxiv.org/abs/2102.11174
- Fast-weight Product Key Memory: https://arxiv.org/abs/2601.00671

The intended contribution of this repository is an auditable CPU protocol:
collision-safe addressing, bounded failure semantics, sparse integer updates,
packed-size accounting, and an explicit in-distribution/OOD benchmark.

## What would be required to approach a serious LLM

A practical follow-up, still cheaper than training from scratch, would be:

1. start from a legal local teacher checkpoint or an existing small model;
2. collect teacher logits/traces offline, without a paid API;
3. distill the common distribution into a 4-bit small core;
4. compile high-surprise, high-value residuals into AURORA memory;
5. use a verifier/gate to invoke the memory only when it improves the prior;
6. evaluate on held-out, contamination-controlled tasks;
7. report quality, bytes, FLOPs, latency, and failure rate together.

Without step 1, equivalent broad knowledge cannot be conjured on this machine.

## Files

- `aurora.py` — implementation and reproducible benchmark.
- `test_aurora.py` — tests for exact replay, honest OOD misses, and collision
  behavior.
- `tiny_lm.cpp` — a real from-scratch character-level causal neural LM with
  truncated BPTT/Adam and an AURORA validation overlay.
- `sweep.py` — a small vocabulary-scaling study for the Python memory compiler.
- `data/tinyshakespeare_200k.txt` — the GitHub-only 200 KB training subset;
  provenance and checksum are in `data/README.md`.

## Real local neural experiment

The second experiment is deliberately a genuine learned model rather than a
lookup-only demo. `tiny_lm.cpp` implements a small Elman causal language model:

```text
h_t = tanh(E[x_t] + W h_{t-1} + b)
logits_t = O h_t + b_o
```

It is trained with full truncated backpropagation through time and separate
Adam state for every parameter tensor. The data is split into 80% train, 10%
calibration, and 10% untouched test. AURORA memory is built from train only;
one scalar memory boost is selected on calibration and the headline score is
reported on test. This is a character LM, not a Transformer and not an
open-domain frontier LLM; its perplexity must not be compared numerically to
BPE-tokenized Sonnet reports.

Build and run:

```bash
c++ -O3 -std=c++17 -Wall -Wextra -pedantic tiny_lm.cpp -o tiny_lm
./tiny_lm data/tinyshakespeare_200k.txt --steps 3000 --seq 64 --batch 2 --hidden 64
```

On the supplied CPU run, the JSON result was:

```text
train tokens                         160,000
calibration tokens                    20,000
test tokens                           20,000
vocabulary                                62
RNN parameters                        12,158
training time                         3.69 s
baseline test loss                    2.19923
baseline character perplexity          9.01808
baseline top-1 accuracy               38.587%
AURORA test loss                      2.05802
AURORA character perplexity            7.83044
AURORA top-1 accuracy                 43.812%
AURORA test memory hits                4,690
compiled sparse entries                8,744
```

On this untouched test split, the overlay reduced loss by about 6.4%,
perplexity by about 13.2%, and increased top-1 accuracy by 5.2 percentage
points. It does not change the trained RNN weights. The memory compiler
observed 48,055 distinct five-character contexts and estimated 22.3x less
packed memory than a dense 2-byte next-character vector for every observed
context. The scalar gate was selected on calibration rather than test, which
avoids using the headline split to tune the result.

The generated sample is printed by the executable. It is visibly more
Shakespeare-like than the untrained/random sample, but qualitative samples are
not a substitute for the reported loss and accuracy.
