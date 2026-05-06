"""
Deterministic benchmark for the BP `_e_index` removal refactor.

Usage
-----
From the repo root:

    /opt/miniconda3/envs/octowaddle/bin/python setup.py build_ext --inplace
    /opt/miniconda3/envs/octowaddle/bin/python benchmarks/bench_bp_refactor.py

The script prints CSV to stdout by default.

Comparing commits
-----------------
Run the same two commands on each commit and capture stdout:

    /opt/miniconda3/envs/octowaddle/bin/python benchmarks/bench_bp_refactor.py > before.csv
    /opt/miniconda3/envs/octowaddle/bin/python benchmarks/bench_bp_refactor.py > after.csv

Then compare:
- `construct_s` for construction time
- `rss_delta_bytes` and `tracemalloc_peak_kb` for construction memory
- `*_ns_per_op` columns for operation timing

Checksums are included so results can be sanity-checked across runs.
"""

import csv
import gc
import os
import platform
import random
import resource
import sys
import time
import tracemalloc

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bp import BP


SEED = 20250217
NS = (256, 1024, 4096, 16384, 32768, 65536)
MODES = ("nested", "flat", "mixed", "random")
BETA = 1 << 15

DEPTH_QUERIES = 512
CLOSE_QUERIES = 512
PARENT_QUERIES = 512
BUCKET_QUERIES = 256
RANGE_QUERIES = 128
MIN_QUERIES = 32


def nested_bits(n):
    return np.array(([1] * n) + ([0] * n), dtype=np.uint8)


def flat_bits(n):
    return np.array([1, 0] * n, dtype=np.uint8)


def mixed_bits(n):
    return np.array([1] + ([1, 0] * (n - 1)) + [0], dtype=np.uint8)


def random_bits(n, seed):
    rng = random.Random(seed)
    bits = []
    opens_used = 0
    closes_used = 0
    balance = 0

    while len(bits) < (2 * n):
        if opens_used == n:
            bit = 0
        elif balance == 0:
            bit = 1
        elif closes_used == n:
            bit = 1
        else:
            bit = rng.randint(0, 1)

        bits.append(bit)
        if bit:
            opens_used += 1
            balance += 1
        else:
            closes_used += 1
            balance -= 1

    return np.array(bits, dtype=np.uint8)


def make_bits(n, mode):
    if mode == "nested":
        return nested_bits(n)
    if mode == "flat":
        return flat_bits(n)
    if mode == "mixed":
        return mixed_bits(n)
    if mode == "random":
        return random_bits(n, SEED + n)
    raise ValueError(f"unknown mode: {mode}")


def make_positions(size, limit, seed):
    if size <= limit:
        return list(range(size))
    rng = random.Random(seed)
    return sorted(rng.sample(range(size), limit))


def make_open_positions(B, limit, seed):
    opens = np.flatnonzero(B == 1).tolist()
    if len(opens) <= limit:
        return opens
    rng = random.Random(seed)
    return sorted(rng.sample(opens, limit))


def matching_maps(B):
    stack = []
    open_to_close = {}
    close_to_open = {}

    for idx, bit in enumerate(B):
        if bit:
            stack.append(idx)
        else:
            open_idx = stack.pop()
            open_to_close[open_idx] = idx
            close_to_open[idx] = open_idx

    return open_to_close, close_to_open


def parent_maps(B):
    stack = []
    open_to_parent = {}
    close_to_parent = {}

    for idx, bit in enumerate(B):
        if bit:
            open_to_parent[idx] = stack[-1] if stack else -1
            stack.append(idx)
        else:
            open_idx = stack.pop()
            close_to_parent[idx] = open_to_parent[open_idx]

    return open_to_parent, close_to_parent


def sample_positions(positions, limit, seed):
    if len(positions) <= limit:
        return sorted(positions)
    rng = random.Random(seed)
    return sorted(rng.sample(positions, limit))


def make_close_bucket_positions(B, beta, limit, seed):
    open_to_close, _ = matching_maps(B)
    same_bucket = []
    cross_bucket = []

    for open_idx, close_idx in open_to_close.items():
        if (open_idx // beta) == (close_idx // beta):
            same_bucket.append(open_idx)
        else:
            cross_bucket.append(open_idx)

    return {
        "same_bucket": sample_positions(same_bucket, limit, seed),
        "cross_bucket": sample_positions(cross_bucket, limit, seed + 1),
    }


def make_parent_positions(size, limit, seed):
    positions = list(range(1, size - 1))
    if len(positions) <= limit:
        return positions
    rng = random.Random(seed)
    return sorted(rng.sample(positions, limit))


def make_parent_bucket_positions(B, beta, limit, seed):
    open_to_parent, close_to_parent = parent_maps(B)
    same_bucket = []
    cross_bucket = []

    for pos in range(1, len(B) - 1):
        if B[pos]:
            parent = open_to_parent[pos]
        else:
            parent = close_to_parent[pos]

        if parent == -1:
            continue

        if (pos // beta) == (parent // beta):
            same_bucket.append(pos)
        else:
            cross_bucket.append(pos)

    return {
        "same_bucket": sample_positions(same_bucket, limit, seed),
        "cross_bucket": sample_positions(cross_bucket, limit, seed + 1),
    }


def make_intervals(size, limit, seed):
    rng = random.Random(seed)
    intervals = []
    widths = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        max(1, size // 32),
        max(1, size // 16),
        max(1, size // 8),
    )

    for idx in range(limit):
        width = widths[idx % len(widths)]
        if width >= size:
            i = 0
            j = size - 1
        else:
            i = rng.randint(0, size - width - 1)
            j = min(size - 1, i + width)
        if i == j:
            j = min(size - 1, i + 1)
        intervals.append((i, j))

    return intervals


def make_minselect_queries(bp, intervals):
    queries = []
    for idx, (i, j) in enumerate(intervals):
        count = bp.mincount(i, j)
        if idx % 3 == 0:
            q = 1
        elif idx % 3 == 1:
            q = count
        else:
            q = count + 1
        queries.append((i, j, q))
    return queries


def make_queries(bp, B, seed_base):
    size = len(B)
    close_bucket_positions = make_close_bucket_positions(
        B, BETA, BUCKET_QUERIES, seed_base + 20
    )
    parent_bucket_positions = make_parent_bucket_positions(
        B, BETA, BUCKET_QUERIES, seed_base + 30
    )

    return {
        "depth": make_positions(size, DEPTH_QUERIES, seed_base + 1),
        "close": make_open_positions(B, CLOSE_QUERIES, seed_base + 2),
        "close_same_bucket": close_bucket_positions["same_bucket"],
        "close_cross_bucket": close_bucket_positions["cross_bucket"],
        "parent": make_parent_positions(size, PARENT_QUERIES, seed_base + 3),
        "parent_same_bucket": parent_bucket_positions["same_bucket"],
        "parent_cross_bucket": parent_bucket_positions["cross_bucket"],
        "rmq": make_intervals(size, RANGE_QUERIES, seed_base + 4),
        "rMq": make_intervals(size, RANGE_QUERIES, seed_base + 5),
        "mincount": make_intervals(size, MIN_QUERIES, seed_base + 6),
        "minselect": None,
    }


def rss_bytes():
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return rss
    return rss * 1024


def measure_construction(B):
    gc.collect()
    rss_before = rss_bytes()
    tracemalloc.start()
    t0 = time.perf_counter()
    bp = BP(B)
    construct_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = rss_bytes()
    return {
        "bp": bp,
        "construct_s": construct_s,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        "tracemalloc_peak_kb": peak / 1024.0,
    }


def time_depth(bp, positions):
    checksum = 0
    t0 = time.perf_counter()
    for pos in positions:
        checksum += int(bp.depth(pos))
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(positions)


def time_close(bp, positions):
    checksum = 0
    t0 = time.perf_counter()
    for pos in positions:
        checksum += int(bp.close(pos))
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(positions)


def time_parent(bp, positions):
    checksum = 0
    t0 = time.perf_counter()
    for pos in positions:
        checksum += int(bp.parent(pos))
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(positions)


def time_rmq(bp, intervals):
    checksum = 0
    t0 = time.perf_counter()
    for i, j in intervals:
        checksum += int(bp.rmq(i, j))
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(intervals)


def time_rMq(bp, intervals):
    checksum = 0
    t0 = time.perf_counter()
    for i, j in intervals:
        checksum += int(bp.rMq(i, j))
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(intervals)


def time_mincount(bp, intervals):
    checksum = 0
    t0 = time.perf_counter()
    for i, j in intervals:
        checksum += int(bp.mincount(i, j))
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(intervals)


def time_minselect(bp, queries):
    checksum = 0
    t0 = time.perf_counter()
    for i, j, q in queries:
        result = bp.minselect(i, j, q)
        checksum += -1 if result is None else int(result)
    elapsed = time.perf_counter() - t0
    return elapsed, checksum, len(queries)


def ns_per_op(elapsed_s, count):
    return (elapsed_s * 1e9) / max(1, count)


def run_case(n, mode):
    B = make_bits(n, mode)
    construct = measure_construction(B)
    bp = construct["bp"]
    seed_base = SEED + (97 * n) + len(mode)
    queries = make_queries(bp, B, seed_base)
    queries["minselect"] = make_minselect_queries(bp, queries["mincount"])

    timings = {}
    checksum = 0

    for name, fn, q in (
        ("depth", time_depth, queries["depth"]),
        ("close", time_close, queries["close"]),
        ("close_same_bucket", time_close, queries["close_same_bucket"]),
        ("close_cross_bucket", time_close, queries["close_cross_bucket"]),
        ("parent", time_parent, queries["parent"]),
        ("parent_same_bucket", time_parent, queries["parent_same_bucket"]),
        ("parent_cross_bucket", time_parent, queries["parent_cross_bucket"]),
        ("rmq", time_rmq, queries["rmq"]),
        ("rMq", time_rMq, queries["rMq"]),
        ("mincount", time_mincount, queries["mincount"]),
        ("minselect", time_minselect, queries["minselect"]),
    ):
        elapsed, partial_checksum, count = fn(bp, q)
        timings[name] = ns_per_op(elapsed, count)
        timings[f"{name}_checksum"] = partial_checksum
        checksum += partial_checksum

    return {
        "n": n,
        "size": len(B),
        "mode": mode,
        "construct_s": f"{construct['construct_s']:.6f}",
        "rss_before_bytes": construct["rss_before_bytes"],
        "rss_after_bytes": construct["rss_after_bytes"],
        "rss_delta_bytes": construct["rss_delta_bytes"],
        "tracemalloc_peak_kb": f"{construct['tracemalloc_peak_kb']:.1f}",
        "depth_ns_per_op": f"{timings['depth']:.1f}",
        "close_ns_per_op": f"{timings['close']:.1f}",
        "close_same_bucket_ns_per_op": f"{timings['close_same_bucket']:.1f}",
        "close_cross_bucket_ns_per_op": f"{timings['close_cross_bucket']:.1f}",
        "parent_ns_per_op": f"{timings['parent']:.1f}",
        "parent_same_bucket_ns_per_op": f"{timings['parent_same_bucket']:.1f}",
        "parent_cross_bucket_ns_per_op": f"{timings['parent_cross_bucket']:.1f}",
        "rmq_ns_per_op": f"{timings['rmq']:.1f}",
        "rMq_ns_per_op": f"{timings['rMq']:.1f}",
        "mincount_ns_per_op": f"{timings['mincount']:.1f}",
        "minselect_ns_per_op": f"{timings['minselect']:.1f}",
        "close_same_bucket_checksum": timings["close_same_bucket_checksum"],
        "close_cross_bucket_checksum": timings["close_cross_bucket_checksum"],
        "parent_same_bucket_checksum": timings["parent_same_bucket_checksum"],
        "parent_cross_bucket_checksum": timings["parent_cross_bucket_checksum"],
        "checksum": checksum,
    }


def main():
    fieldnames = [
        "n",
        "size",
        "mode",
        "construct_s",
        "rss_before_bytes",
        "rss_after_bytes",
        "rss_delta_bytes",
        "tracemalloc_peak_kb",
        "depth_ns_per_op",
        "close_ns_per_op",
        "close_same_bucket_ns_per_op",
        "close_cross_bucket_ns_per_op",
        "parent_ns_per_op",
        "parent_same_bucket_ns_per_op",
        "parent_cross_bucket_ns_per_op",
        "rmq_ns_per_op",
        "rMq_ns_per_op",
        "mincount_ns_per_op",
        "minselect_ns_per_op",
        "close_same_bucket_checksum",
        "close_cross_bucket_checksum",
        "parent_same_bucket_checksum",
        "parent_cross_bucket_checksum",
        "checksum",
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()

    for n in NS:
        for mode in MODES:
            writer.writerow(run_case(n, mode))


if __name__ == "__main__":
    main()
