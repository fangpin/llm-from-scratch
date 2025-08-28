# Benchmarks

A benchmarking script is provided in `kernel/benchmarking_model.py` to measure the performance of the Transformer model. This script can be used to evaluate the training and inference speed.

Benchmark the model:

```bash
uv run -m kernel.bench_mark.bench_mark_model

```
output:

```bash

Time per iteration: 53.742 ms, total time: 5374.178 ms
```

Benchmark the pytorch scaled-dot-product attention:

```bash
uv run -m kernel.bench_mark.bench_mark_atten
```
output:
```bash
dtype: torch.float32, d_model:     16, seq_len:    256: Time per iteration: 0.235 ms, total time: 23.507 ms
dtype: torch.float32, d_model:     16, seq_len:   1024: Time per iteration: 2.795 ms, total time: 279.508 ms
dtype: torch.float32, d_model:     16, seq_len:   4096: Time per iteration: 43.079 ms, total time: 4307.860 ms
dtype: torch.float32, d_model:     32, seq_len:    256: Time per iteration: 0.243 ms, total time: 24.307 ms
dtype: torch.float32, d_model:     32, seq_len:   1024: Time per iteration: 2.903 ms, total time: 290.327 ms
dtype: torch.float32, d_model:     32, seq_len:   4096: Time per iteration: 44.843 ms, total time: 4484.318 ms
dtype: torch.float32, d_model:     64, seq_len:    256: Time per iteration: 0.257 ms, total time: 25.686 ms
dtype: torch.float32, d_model:     64, seq_len:   1024: Time per iteration: 3.130 ms, total time: 313.000 ms
dtype: torch.float32, d_model:     64, seq_len:   4096: Time per iteration: 48.445 ms, total time: 4844.524 ms
dtype: torch.float32, d_model:    128, seq_len:    256: Time per iteration: 0.293 ms, total time: 29.329 ms
dtype: torch.float32, d_model:    128, seq_len:   1024: Time per iteration: 3.563 ms, total time: 356.268 ms
dtype: torch.float32, d_model:    128, seq_len:   4096: Time per iteration: 55.330 ms, total time: 5532.962 ms
dtype: torch.bfloat16, d_model:     16, seq_len:    256: Time per iteration: 0.244 ms, total time: 24.431 ms
dtype: torch.bfloat16, d_model:     16, seq_len:   1024: Time per iteration: 1.420 ms, total time: 142.000 ms
dtype: torch.bfloat16, d_model:     16, seq_len:   4096: Time per iteration: 19.901 ms, total time: 1990.081 ms
dtype: torch.bfloat16, d_model:     32, seq_len:    256: Time per iteration: 0.228 ms, total time: 22.800 ms
dtype: torch.bfloat16, d_model:     32, seq_len:   1024: Time per iteration: 1.423 ms, total time: 142.288 ms
dtype: torch.bfloat16, d_model:     32, seq_len:   4096: Time per iteration: 19.912 ms, total time: 1991.237 ms
dtype: torch.bfloat16, d_model:     64, seq_len:    256: Time per iteration: 0.229 ms, total time: 22.945 ms
dtype: torch.bfloat16, d_model:     64, seq_len:   1024: Time per iteration: 1.376 ms, total time: 137.563 ms
dtype: torch.bfloat16, d_model:     64, seq_len:   4096: Time per iteration: 20.446 ms, total time: 2044.641 ms
dtype: torch.bfloat16, d_model:    128, seq_len:    256: Time per iteration: 0.225 ms, total time: 22.488 ms
dtype: torch.bfloat16, d_model:    128, seq_len:   1024: Time per iteration: 1.428 ms, total time: 142.778 ms
dtype: torch.bfloat16, d_model:    128, seq_len:   4096: Time per iteration: 21.075 ms, total time: 2107.540 ms
```

Benchmark the pytorch JIT compiled scaled-dot-product attention:

```bash
uv run -m kernel.bench_mark.bench_mark_atten_jit
```
output:
```bash
dtype: torch.float32, d_model:     16, seq_len:    256: Time per iteration: 1.674 ms, total time: 167.437 ms
dtype: torch.float32, d_model:     16, seq_len:   1024: Time per iteration: 2.979 ms, total time: 297.851 ms
dtype: torch.float32, d_model:     16, seq_len:   4096: Time per iteration: 23.964 ms, total time: 2396.361 ms
dtype: torch.float32, d_model:     32, seq_len:    256: Time per iteration: 1.072 ms, total time: 107.243 ms
dtype: torch.float32, d_model:     32, seq_len:   1024: Time per iteration: 2.905 ms, total time: 290.463 ms
dtype: torch.float32, d_model:     32, seq_len:   4096: Time per iteration: 44.846 ms, total time: 4484.633 ms
dtype: torch.float32, d_model:     64, seq_len:    256: Time per iteration: 0.258 ms, total time: 25.767 ms
dtype: torch.float32, d_model:     64, seq_len:   1024: Time per iteration: 3.132 ms, total time: 313.244 ms
dtype: torch.float32, d_model:     64, seq_len:   4096: Time per iteration: 48.441 ms, total time: 4844.149 ms
dtype: torch.float32, d_model:    128, seq_len:    256: Time per iteration: 0.294 ms, total time: 29.399 ms
dtype: torch.float32, d_model:    128, seq_len:   1024: Time per iteration: 3.571 ms, total time: 357.092 ms
dtype: torch.float32, d_model:    128, seq_len:   4096: Time per iteration: 55.339 ms, total time: 5533.872 ms
dtype: torch.bfloat16, d_model:     16, seq_len:    256: Time per iteration: 0.289 ms, total time: 28.885 ms
dtype: torch.bfloat16, d_model:     16, seq_len:   1024: Time per iteration: 1.421 ms, total time: 142.089 ms
dtype: torch.bfloat16, d_model:     16, seq_len:   4096: Time per iteration: 19.907 ms, total time: 1990.715 ms
dtype: torch.bfloat16, d_model:     32, seq_len:    256: Time per iteration: 0.259 ms, total time: 25.888 ms
dtype: torch.bfloat16, d_model:     32, seq_len:   1024: Time per iteration: 1.424 ms, total time: 142.393 ms
dtype: torch.bfloat16, d_model:     32, seq_len:   4096: Time per iteration: 19.920 ms, total time: 1991.968 ms
dtype: torch.bfloat16, d_model:     64, seq_len:    256: Time per iteration: 0.268 ms, total time: 26.839 ms
dtype: torch.bfloat16, d_model:     64, seq_len:   1024: Time per iteration: 1.374 ms, total time: 137.391 ms
dtype: torch.bfloat16, d_model:     64, seq_len:   4096: Time per iteration: 20.454 ms, total time: 2045.394 ms
dtype: torch.bfloat16, d_model:    128, seq_len:    256: Time per iteration: 0.266 ms, total time: 26.551 ms
dtype: torch.bfloat16, d_model:    128, seq_len:   1024: Time per iteration: 1.428 ms, total time: 142.778 ms
dtype: torch.bfloat16, d_model:    128, seq_len:   4096: Time per iteration: 21.078 ms, total time: 2107.774 ms
```