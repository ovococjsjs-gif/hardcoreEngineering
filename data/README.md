# Local benchmark data

`tinyshakespeare_200k.txt` is the first 200,000 bytes of the Tiny Shakespeare
corpus stored in the GitHub repository:

- source repository: https://github.com/karpathy/char-rnn
- source revision used: `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`
- source path: `data/tinyshakespeare/input.txt`
- source size: 1,115,394 bytes
- local subset size: 200,000 bytes
- local SHA-256: `74990c048a8dbd1535178eeecbcbc090d6509224e31b5387ef4887e354922a30`

The subset is kept small enough for a reproducible local CPU experiment. The
first 80% is used for training, the next 10% for gate calibration, and the last
10% is an untouched test split. The model is character-level, so the measured
perplexity is not comparable numerically with word- or BPE-tokenized LLM
reports.
