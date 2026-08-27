#!/usr/bin/env python3
"""Generate an honest measured-vs-projected AURORA scaling report.

Only Python's standard library is used.  The generated SVG is intentionally
plain so it can be opened without a plotting package.
"""

from __future__ import annotations

from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent

# Values from the reproducible run recorded in README.md.  These are not
# re-estimated from a model-size story.
BASE_TEST_LOSS = 2.19923
HYBRID_TEST_LOSS = 2.05802
BASE_PPL = 9.01808
HYBRID_PPL = 7.83044
BASE_ACC = 0.385869
HYBRID_ACC = 0.438122
MEMORY_HITS = 4690
MEMORY_ENTRIES = 8744
OBSERVED_CONTEXTS = 48055
DENSE_BYTES = 6343384
SPARSE_BYTES = 284224
TRAIN_SECONDS = 3.69032
RNN_PARAMS = 12158
VOCAB = 62

# AURORA's local memory operation on the previous Python microbenchmark used
# one sparse candidate rather than a V-wide scan.  It is an operation count,
# not a wall-clock claim.
DEMO_V = 2048
DEMO_K = 1
COMPONENT_SPEEDUP = DEMO_V / DEMO_K
MEMORY_RATIO = DENSE_BYTES / SPARSE_BYTES
LOSS_REDUCTION = 1.0 - HYBRID_TEST_LOSS / BASE_TEST_LOSS
PPL_REDUCTION = 1.0 - HYBRID_PPL / BASE_PPL
ACC_GAIN_PP = (HYBRID_ACC - BASE_ACC) * 100.0


def amdahl(fraction: float, component_speedup: float) -> float:
    """End-to-end speedup when only a fraction is accelerated."""
    return 1.0 / ((1.0 - fraction) + fraction / component_speedup)


def fmt_x(value: float) -> str:
    return f"{value:,.2f}x"


def svg_bar(x: float, y: float, width: float, height: float, value: float,
            max_value: float, color: str, label: str, value_label: str) -> str:
    # Log scale keeps 1x, 22x and 2048x visible in one chart.
    lo = 0.0
    hi = math.log10(max_value)
    scaled = (math.log10(max(value, 1.0)) - lo) / max(hi, 1e-9)
    w = max(2.0, width * scaled)
    return (
        f'<text x="{x:.1f}" y="{y + 15:.1f}" class="label">{label}</text>'
        f'<rect x="{x + 205:.1f}" y="{y:.1f}" width="{w:.1f}" '
        f'height="{height:.1f}" rx="4" fill="{color}"/>'
        f'<text x="{x + 215 + w:.1f}" y="{y + 15:.1f}" class="value">'
        f'{value_label}</text>'
    )


def make_svg() -> str:
    width, height = 1200, 790
    parts = [
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
.title {{ font: 700 24px sans-serif; fill: #172033; }}
.subtitle {{ font: 14px sans-serif; fill: #536174; }}
.panel {{ fill: #f7f9fc; stroke: #d7deea; stroke-width: 1; }}
.heading {{ font: 700 17px sans-serif; fill: #172033; }}
.label {{ font: 13px sans-serif; fill: #334155; }}
.value {{ font: 700 13px sans-serif; fill: #172033; }}
.axis {{ font: 11px sans-serif; fill: #64748b; }}
.note {{ font: 12px sans-serif; fill: #536174; }}
.grid {{ stroke: #d7deea; stroke-width: 1; stroke-dasharray: 3 4; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="36" y="38" class="title">AURORA-PRM: measured result vs scalable component model</text>
<text x="36" y="62" class="subtitle">Log bars are ratios, not a promise of end-to-end frontier-LLM speed. Measured on the local CPU prototype.</text>

<rect x="30" y="82" width="550" height="292" rx="8" class="panel"/>
<text x="50" y="112" class="heading">1. What was actually measured</text>
<text x="50" y="133" class="note">The current RNN was trained once; AURORA was compiled after training.</text>
'''
    ]
    bars = [
        (COMPONENT_SPEEDUP, '#2563eb', 'Sparse candidate work', fmt_x(COMPONENT_SPEEDUP)),
        (MEMORY_RATIO, '#0891b2', 'Packed contextual memory', fmt_x(MEMORY_RATIO)),
        (HYBRID_PPL / BASE_PPL, '#16a34a', 'Perplexity ratio (lower is better)', fmt_x(BASE_PPL / HYBRID_PPL)),
        (1.0, '#64748b', 'Core training wall-clock speedup', '1.00x / not accelerated'),
    ]
    y = 155
    for value, color, label, value_label in bars:
        parts.append(svg_bar(50, y, 340, 24, value, COMPONENT_SPEEDUP, color, label, value_label))
        y += 42
    parts += [
        '<text x="50" y="344" class="note">Training speedup: 1.00x is the honest result for this version.</text>',
        '<rect x="610" y="82" width="560" height="292" rx="8" class="panel"/>',
        '<text x="630" y="112" class="heading">2. Amdahl: end-to-end speedup</text>',
        '<text x="630" y="133" class="note">Only the accelerated fraction f benefits from a 2,048x local operation reduction.</text>',
    ]
    fractions = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    max_s = max(amdahl(f, COMPONENT_SPEEDUP) for f in fractions)
    x0, y0, chart_w, chart_h = 650, 160, 465, 165
    # axes and grid
    for tick in [1, 2, 5, 10, 20, 50, 100]:
        if tick > max_s * 1.15:
            continue
        yy = y0 + chart_h - (math.log10(tick) / math.log10(max(100, max_s * 1.1))) * chart_h
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + chart_w}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 28}" y="{yy + 4:.1f}" class="axis">{tick}x</text>')
    points = []
    for i, f in enumerate(fractions):
        s = amdahl(f, COMPONENT_SPEEDUP)
        xx = x0 + (i / (len(fractions) - 1)) * chart_w
        yy = y0 + chart_h - (math.log10(s) / math.log10(max(100, max_s * 1.1))) * chart_h
        points.append(f'{xx:.1f},{yy:.1f}')
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="#7c3aed"/>')
        parts.append(f'<text x="{xx - 14:.1f}" y="{y0 + chart_h + 20}" class="axis">{int(f*100)}%</text>')
        parts.append(f'<text x="{xx - 15:.1f}" y="{yy - 10:.1f}" class="value">{s:.2f}x</text>')
    parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#7c3aed" stroke-width="3"/>')
    parts += [
        f'<text x="650" y="350" class="note">At f=90%: {amdahl(.90, COMPONENT_SPEEDUP):.2f}x end-to-end.</text>',
        '<text x="650" y="366" class="note">To reach 1,000x total, over 99.95% of all baseline work would have to be replaced.</text>',
        '<rect x="30" y="397" width="1140" height="356" rx="8" class="panel"/>',
        '<text x="50" y="428" class="heading">3. Projection for larger vocabularies / LLM-like systems</text>',
        '<text x="50" y="449" class="note">This is the AURORA component only. Core weights, reasoning and data costs remain.</text>',
    ]
    # table
    cols = [50, 160, 280, 425, 610, 825]
    headers = ['archetype', 'core params', 'vocab V', 'sparse k', 'component V/k', '4-bit core*']
    for x, h in zip(cols, headers):
        parts.append(f'<text x="{x}" y="478" class="value">{h}</text>')
    rows = [
        ('small', '7B', '32K', '1', '32,000x', '3.5 GB'),
        ('medium', '13B', '32K', '1', '32,000x', '6.5 GB'),
        ('large', '70B', '128K', '4', '32,000x', '35 GB'),
        ('frontier', '400B', '128K', '4', '32,000x', '200 GB'),
    ]
    yy = 505
    for row in rows:
        for x, text in zip(cols, row):
            parts.append(f'<text x="{x}" y="{yy}" class="label">{text}</text>')
        yy += 30
    parts += [
        '<text x="50" y="640" class="note">*4-bit core size is only P×0.5 bytes, excluding scales, KV cache and runtime overhead.</text>',
        '<text x="50" y="660" class="note">AURORA does not shrink these core weights. It can shrink a separate dense V-wide contextual memory.</text>',
        '<text x="50" y="686" class="note">Raw one-entry storage estimate: dense 2V bytes vs sparse ~25 bytes; real benefit depends on hit rate and bandwidth.</text>',
        '<text x="50" y="712" class="note">Use the report tables for formulas and the exact distinction between measured and projected numbers.</text>',
        '</svg>'
    ]
    return '\n'.join(parts)


def write_report() -> None:
    report = f'''# AURORA scaling report: measured vs projected

Generated by `scaling_report.py`; no plotting dependency is required.

## Headline answer

| Quantity | Honest result for the current implementation |
|---|---:|
| Core RNN training wall-clock speedup | **1.00x** — AURORA is compiled after training; training was not accelerated |
| Earlier Python memory-compiler wall-clock | AURORA was about **0.85x** the dense reference in that microbenchmark; hashing overhead dominated |
| Sparse candidate work on an exact memory path | **2,048x less** (`V=2048`, `k=1`); this is not full-model latency |
| Packed contextual memory | **22.32x less** in the real C++ Shakespeare run |
| Test perplexity | 9.01808 → 7.83044, **13.17% lower** |
| Test top-1 accuracy | 38.5869% → 43.8122%, **+5.23 percentage points** |
| Dollar cost | **Not measured**; no cloud price or power meter was used |

The model-side sparse operation estimate from the earlier Python benchmark was
40,960,000 dense output operations versus 20,000 sparse updates. That is a
**2,048x operation-count ratio**, not a measured 2,048x training-time speedup.

## What was measured

The final local neural run used 160,000 training characters, a 20,000-character
calibration split, and a separate 20,000-character test split. It trained a
12,158-parameter character RNN in 3.69 seconds. The memory was constructed only
from the training split; boost 4.0 was selected on calibration and then frozen
for test.

The test memory had {MEMORY_HITS:,} hits, {MEMORY_ENTRIES:,} compiled entries,
and {OBSERVED_CONTEXTS:,} contexts observed during compilation. This is why the
quality improvement is conditional: only 4,690 of 19,999 test transitions used
the memory path.

## Why the ratio is not the end-to-end ratio

Let `S_c` be the accelerated component speedup and `f` the fraction of total
baseline time spent in that component. Amdahl's law gives:

```text
S_total = 1 / ((1 - f) + f / S_c)
```

For `S_c = 2048`:

| Accelerated fraction f | End-to-end speedup |
|---:|---:|
'''
    for f in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        report += f'| {f:.0%} | {amdahl(f, COMPONENT_SPEEDUP):.3f}x |\n'
    report += f'''
Even a 90%-sparse redesign would be only {amdahl(.90, COMPONENT_SPEEDUP):.2f}x
end-to-end. A 1,000x total speedup would require approximately
{(1 - 1/1000) / (1 - 1/COMPONENT_SPEEDUP):.4%} of all baseline work to be this
specific replaced component. The remaining neural core, memory bandwidth,
communication, sampling and I/O cannot be ignored.

## Large-scale projection

The following numbers are a transparent scaling model, not measurements of
7B/70B/400B models:

| Archetype | Core params | V | active sparse k | AURORA component ratio V/k | idealized 4-bit core weights |
|---|---:|---:|---:|---:|---:|
| small | 7B | 32K | 1 | 32,000x | 3.5 GB |
| medium | 13B | 32K | 1 | 32,000x | 6.5 GB |
| large | 70B | 128K | 4 | 32,000x | 35 GB |
| frontier | 400B | 128K | 4 | 32,000x | 200 GB |

The 4-bit column is just `parameters × 0.5 bytes`; it excludes quantization
scales, activations, KV cache, runtime buffers and sharding. AURORA does not
reduce the core weights. It can reduce a separate dense contextual table: a
single dense `uint16` V-wide row costs `2V` bytes, whereas the current packed
sparse entry is roughly 25 bytes before system-level indexes.

The SVG visualization is [scaling.svg](scaling.svg).

## What the current result does and does not establish

**Established by this repository:**

- an actual local neural LM trains from GitHub data;
- the held-out test split is separate from gate calibration;
- the sparse overlay improves the reported character-level metrics on this split;
- sparse exact retrieval and collision-safe fallback work;
- storage and candidate-operation ratios are reproducible.

**Not established:**

- Sonnet-level quality;
- general reasoning or multilingual ability;
- a 10x/100x/1000x reduction in full LLM training wall-clock;
- a dollar cost reduction, since no paid hardware/API was used;
- a speedup for an arbitrary Transformer whose dense output projection has not
  been replaced by a compatible adaptive-softmax/indexing design.
'''
    (ROOT / 'SCALING_REPORT.md').write_text(report, encoding='utf-8')
    (ROOT / 'scaling.svg').write_text(make_svg(), encoding='utf-8')


if __name__ == '__main__':
    write_report()
    print('wrote SCALING_REPORT.md and scaling.svg')
''