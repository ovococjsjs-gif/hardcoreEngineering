# Second system-level attempt: CIRRUS

## Scope and honest target

The earlier AURORA experiment changed only a contextual memory overlay. This
experiment changes the learned core and the inference representation too. It
still cannot honestly promise a general Sonnet-level model on a two-core, 4 GiB
host without a strong teacher or a large information source. The point here is
to measure the full trade-off instead of calling a local component ratio a
frontier-model result.

## Architecture

CIRRUS means **Compressed Interleaved Recurrent Routing with Quantized and
Uncertainty-gated Sparse memory**. Its learned causal core is:

```text
r_t = B h_(t-1)                    B: rank x hidden
h_t = tanh(E[x_t] + A r_t + b)      A: hidden x rank
u_t = C h_t                        C: output_rank x hidden
logits_t = D u_t + b_o              D: vocab x output_rank
```

It therefore replaces a dense recurrent matrix and dense output matrix with
factorized matrices. After training, all tensors are symmetrically quantized
to int8. AURORA-style memory is compiled from the training split and can add a
confidence-weighted correction for high-confidence five-character contexts.
The gate scalar is selected on calibration, then frozen before the final test.

The forward implementation has a streaming `step` path, so evaluation does not
recompute the complete prefix for every token. Training uses full truncated BPTT
through `A`, `B`, `C`, and `D`, with independent Adam states.

## Reproducible commands

```bash
c++ -O3 -std=c++17 -Wall -Wextra -pedantic cirrus_lm.cpp -o cirrus_lm
./cirrus_lm data/tinyshakespeare_200k.txt \
  --steps 5000 --seq 64 --batch 2 --hidden 64 \
  --rank 16 --output-rank 16 --memory-order 5 \
  --min-seen 2 --confidence 0.8 --boost 2.0
```

The baseline comparison uses the existing full-rank `tiny_lm.cpp` with the same
5,000 steps, sequence length, batch size, hidden size, data split and seed.

## Measured comparison

| quantity | full-rank TinyRNN + AURORA | CIRRUS rank16/16 + int8 + AURORA |
|---|---:|---:|
| train wall-clock | 6.30 s | 4.01 s |
| training speed ratio | 1.00× | 1.57× faster |
| learned parameters | 12,158 | 8,158 |
| parameter count | 1.00× | 0.671× |
| float32 weight bytes | 48,632 theoretical | 32,632 |
| quantized weight bytes | not implemented in baseline | 8,186 |
| test loss | 2.00235 | 2.13208 |
| test perplexity | 7.40646 | 8.43238 |
| test top-1 | 45.0873% | 43.4622% |
| contextual memory bytes | 284,224 | 284,224 |

The CIRRUS core is therefore about 1.57× faster to train and has 33% fewer
parameters, while the int8 core weight artifact is about 5.94× smaller than a
float32 baseline core. However, the final hybrid perplexity is about 13.8%
worse than the full-rank baseline hybrid. This is a real quality loss, not a
successful lossless compression result.

The int8 quantization itself was nearly lossless **within CIRRUS**:

```text
float CIRRUS PPL: 9.83145
int8  CIRRUS PPL: 9.84242
```

That is about a 0.11% relative change in this experiment. The main loss comes
from low-rank factorization, not from int8 quantization.

## What this teaches us

1. Quantization is a credible size reduction route; it did not materially harm
   this tiny model's test score.
2. Low-rank recurrent/output factorization can reduce training work, but the
   rank budget is a real capacity bottleneck.
3. A memory overlay improves both models, but it does not replace semantic
   computation. CIRRUS + memory remains below full-rank + memory.
4. On this CPU, unoptimized Python hashing and naive scalar quantized kernels
   can erase theoretical savings. Kernel implementation matters.
5. The correct next compression experiment is teacher-guided residual
   distillation: retain the low-rank/int8 core, but compile residual corrections
   from a stronger local teacher and use a verifier. That is still conditional
   on having the teacher; it cannot create frontier knowledge from nothing.

## Full-system scaling equation

If a change accelerates only fraction `f` of a baseline, and that fraction has
component speedup `S_c`, then:

```text
S_total = 1 / ((1 - f) + f / S_c)
```

For the earlier `S_c = 2048` candidate-work ratio:

| f | total speedup |
|---:|---:|
| 10% | 1.111× |
| 25% | 1.333× |
| 50% | 1.999× |
| 75% | 3.994× |
| 90% | 9.956× |
| 99% | 95.389× |

A 1,000× whole-system speedup would require replacing 99.9488% of all
baseline work. That is not a realistic consequence of a sparse lookup alone.

## Research-grade score for this attempt

- literal Sonnet-parity goal: **2/10**;
- end-to-end code change: **7/10**;
- empirical honesty: **9/10**;
- experimental rigor: **6/10**;
- demonstrated quality/size trade-off: **6/10**;
- overall: **6/10**.

The score is higher than the first attempt for changing the core, adding a real
int8 forward path and streaming evaluation, but it is not high because the
quality loss remains and no whole-model dollar benchmark exists.
