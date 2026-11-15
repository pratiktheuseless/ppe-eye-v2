"""Latency comparison vs literature targets: python -m tests.benchmark_eval sample.jpg"""
import sys
from models.train_benchmark import benchmark

if __name__ == "__main__":
    benchmark(sys.argv[1])
