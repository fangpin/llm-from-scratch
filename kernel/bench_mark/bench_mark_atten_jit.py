from .bench_mark_atten import benchmark_attention

if __name__ == "__main__":
    benchmark_attention(jit=True)