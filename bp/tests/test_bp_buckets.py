import random

import numpy as np

import bp.tests.test_bp_cy as tbc


SEED = 20250217
INT_MAX = (1 << 31) - 1


def nested_bits(n):
    return np.array(([1] * n) + ([0] * n), dtype=np.uint8)


def flat_bits(n):
    return np.array([1, 0] * n, dtype=np.uint8)


def mixed_bits(n):
    if n == 1:
        return np.array([1, 0], dtype=np.uint8)
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


PATTERNS = {
    "nested": nested_bits,
    "flat": flat_bits,
    "mixed": mixed_bits,
    "random": lambda n: random_bits(n, SEED + n),
}


def bucket_reference(B, beta):
    n_buckets = max(1, (len(B) + beta - 1) // beta)
    bucket_e = []
    bucket_m = []
    bucket_M = []
    excess = 0

    for bucket_idx in range(n_buckets):
        start = bucket_idx * beta
        end = min((bucket_idx + 1) * beta, len(B))
        bucket_min = None
        bucket_max = None

        for pos in range(start, end):
            excess += -1 + (2 * int(B[pos]))
            if bucket_min is None or excess < bucket_min:
                bucket_min = excess
            if bucket_max is None or excess > bucket_max:
                bucket_max = excess

        bucket_e.append(excess)
        bucket_m.append(bucket_min)
        bucket_M.append(bucket_max)

    return (
        n_buckets,
        np.array(bucket_e, dtype=np.intp),
        np.array(bucket_m, dtype=np.intp),
        np.array(bucket_M, dtype=np.intp),
    )


def bucket_tree_reference(bucket_m, bucket_M):
    base = 1
    while base < len(bucket_m):
        base *= 2

    tree_m = np.full(2 * base, INT_MAX, dtype=np.intp)
    tree_M = np.zeros(2 * base, dtype=np.intp)

    for idx in range(len(bucket_m)):
        tree_m[base + idx] = bucket_m[idx]
        tree_M[base + idx] = bucket_M[idx]

    for node in range(base - 1, 0, -1):
        tree_m[node] = min(tree_m[2 * node], tree_m[(2 * node) + 1])
        tree_M[node] = max(tree_M[2 * node], tree_M[(2 * node) + 1])

    return base, tree_m, tree_M


def assert_bucket_summaries(B, expected_n_buckets):
    (beta,
     n_buckets,
     bucket_tree_base,
     bucket_e,
     bucket_m,
     bucket_M,
     bucket_tree_m,
     bucket_tree_M) = tbc.get_bucket_summaries(B)
    ref_n_buckets, ref_e, ref_m, ref_M = bucket_reference(B, beta)
    ref_tree_base, ref_tree_m, ref_tree_M = bucket_tree_reference(ref_m, ref_M)

    assert beta == (1 << 15)
    assert n_buckets == expected_n_buckets
    assert n_buckets == ref_n_buckets
    assert bucket_tree_base >= n_buckets
    assert bucket_tree_base == ref_tree_base
    assert bucket_tree_base & (bucket_tree_base - 1) == 0
    np.testing.assert_array_equal(bucket_e, ref_e)
    np.testing.assert_array_equal(bucket_m, ref_m)
    np.testing.assert_array_equal(bucket_M, ref_M)
    np.testing.assert_array_equal(bucket_tree_m, ref_tree_m)
    np.testing.assert_array_equal(bucket_tree_M, ref_tree_M)

    for idx in range(n_buckets):
        assert bucket_tree_m[bucket_tree_base + idx] == bucket_m[idx]
        assert bucket_tree_M[bucket_tree_base + idx] == bucket_M[idx]

    for idx in range(n_buckets, bucket_tree_base):
        assert bucket_tree_m[bucket_tree_base + idx] == INT_MAX
        assert bucket_tree_M[bucket_tree_base + idx] == 0

    for node in range(bucket_tree_base - 1, 0, -1):
        left = 2 * node
        right = left + 1
        assert bucket_tree_m[node] == min(bucket_tree_m[left], bucket_tree_m[right])
        assert bucket_tree_M[node] == max(bucket_tree_M[left], bucket_tree_M[right])


def test_bucket_summaries_smaller_than_beta():
    for builder in PATTERNS.values():
        assert_bucket_summaries(builder(32), 1)


def test_bucket_summaries_exactly_one_bucket():
    n = (1 << 15) // 2
    for builder in PATTERNS.values():
        assert_bucket_summaries(builder(n), 1)


def test_bucket_summaries_final_partial_bucket():
    n = ((1 << 15) + 2) // 2
    for builder in PATTERNS.values():
        assert_bucket_summaries(builder(n), 2)
