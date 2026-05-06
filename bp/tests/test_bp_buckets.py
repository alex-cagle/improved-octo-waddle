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


def rmm_block_size(size):
    if size <= 2:
        return 1
    import math
    return max(1, int(math.ceil(math.log(size) * math.log(math.log(size)))))


def bucket_find_first_ref(bucket_m, bucket_M, lo_bucket, hi_bucket, target):
    if hi_bucket < 0 or lo_bucket >= len(bucket_m):
        return -1

    lo_bucket = max(0, lo_bucket)
    hi_bucket = min(len(bucket_m) - 1, hi_bucket)
    if lo_bucket > hi_bucket:
        return -1

    for bucket_idx in range(lo_bucket, hi_bucket + 1):
        if bucket_m[bucket_idx] <= target <= bucket_M[bucket_idx]:
            return bucket_idx

    return -1


def bucket_find_last_ref(bucket_m, bucket_M, lo_bucket, hi_bucket, target):
    if hi_bucket < 0 or lo_bucket >= len(bucket_m):
        return -1

    lo_bucket = max(0, lo_bucket)
    hi_bucket = min(len(bucket_m) - 1, hi_bucket)
    if lo_bucket > hi_bucket:
        return -1

    for bucket_idx in range(hi_bucket, lo_bucket - 1, -1):
        if bucket_m[bucket_idx] <= target <= bucket_M[bucket_idx]:
            return bucket_idx

    return -1


def excess_values(B):
    excess = 0
    values = []

    for bit in B:
        excess += -1 + (2 * int(bit))
        values.append(excess)

    return values


def fwdsearch_in_range_ref(B, lo, hi, target):
    excess = excess_values(B)
    for pos in range(lo, hi + 1):
        if excess[pos] == target:
            return pos
    return -1


def bwdsearch_in_range_ref(B, lo, hi, target):
    excess = excess_values(B)
    for pos in range(hi, lo - 1, -1):
        if excess[pos] == target:
            return pos
    return -1


def assert_range_query(B, lo, hi, target):
    exp_fwd = fwdsearch_in_range_ref(B, lo, hi, target)
    exp_bwd = bwdsearch_in_range_ref(B, lo, hi, target)
    obs_fwd, obs_bwd = tbc.get_range_search_results(B, lo, hi, target)
    assert obs_fwd == exp_fwd
    assert obs_bwd == exp_bwd


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


def assert_bucket_tree_searches(B):
    (beta,
     n_buckets,
     bucket_tree_base,
     bucket_e,
     bucket_m,
     bucket_M,
     bucket_tree_m,
     bucket_tree_M) = tbc.get_bucket_summaries(B)

    del beta, bucket_tree_m, bucket_tree_M, bucket_e

    hi_with_padding = bucket_tree_base - 1
    ranges = []
    for lo_bucket in range(-1, n_buckets + 1):
        for hi_bucket in range(lo_bucket, hi_with_padding + 1):
            ranges.append((lo_bucket, hi_bucket))

    max_target = int(bucket_M.max()) if len(bucket_M) else 0
    targets = [-1, 0, max_target + 1]
    for bucket_idx in range(n_buckets):
        targets.extend(
            [
                int(bucket_m[bucket_idx]),
                int(bucket_M[bucket_idx]),
                int((bucket_m[bucket_idx] + bucket_M[bucket_idx]) // 2),
            ]
        )

    for lo_bucket, hi_bucket in ranges:
        for target in sorted(set(targets)):
            exp_first = bucket_find_first_ref(bucket_m, bucket_M, lo_bucket,
                                              hi_bucket, target)
            exp_last = bucket_find_last_ref(bucket_m, bucket_M, lo_bucket,
                                            hi_bucket, target)
            obs_first, obs_last = tbc.get_bucket_tree_search_results(
                B, lo_bucket, hi_bucket, target
            )
            assert obs_first == exp_first
            assert obs_last == exp_last


def test_bucket_tree_searches_one_bucket():
    for builder in PATTERNS.values():
        assert_bucket_tree_searches(builder(32))


def test_bucket_tree_searches_multiple_buckets():
    n = ((3 * (1 << 15)) + 50) // 2
    for builder in PATTERNS.values():
        assert_bucket_tree_searches(builder(n))


def test_bucket_tree_searches_final_partial_bucket():
    n = ((2 * (1 << 15)) + 14) // 2
    for builder in PATTERNS.values():
        assert_bucket_tree_searches(builder(n))


def assert_range_searches(B):
    size = len(B)
    beta = 1 << 15
    b = rmm_block_size(size)
    excess = excess_values(B)
    queries = [
        (0, 0),
        (0, min(size - 1, 5)),
        (max(0, (size // 2) - 1), min(size - 1, (size // 2) + 1)),
        (0, size - 1),
    ]

    if size > 20:
        queries.append((10, min(size - 1, 20)))

    if size > b + 4:
        queries.append((b - 2, min(size - 1, b + 2)))

    if size > (3 * b) + 4:
        queries.append((b - 2, min(size - 1, (3 * b) + 2)))
        queries.append((b + 1, min(size - 1, (4 * b) + 1)))

    if size > beta + 10:
        queries.append((beta - 5, beta + 5))

    if size > (2 * beta):
        queries.append((beta - 5, min(size - 1, (2 * beta) + 5)))

    if size > beta:
        queries.append((max(0, size - 12), size - 1))

    seen = set()
    deduped_queries = []
    for lo, hi in queries:
        key = (lo, hi)
        if key not in seen:
            seen.add(key)
            deduped_queries.append(key)

    for lo, hi in deduped_queries:
        targets = {
            excess[lo],
            excess[hi],
            excess[(lo + hi) // 2],
            max(excess) + 1,
        }

        if hi > lo:
            targets.add(excess[min(hi, lo + 1)])
            targets.add(excess[max(lo, hi - 1)])

        for target in sorted(targets):
            assert_range_query(B, lo, hi, target)


def test_range_searches_crafted_block_cases():
    n = ((5 * (1 << 15)) + 50) // 2
    B = PATTERNS["nested"](n)
    b = rmm_block_size(len(B))
    excess = excess_values(B)

    cases = [
        (0, min(len(B) - 1, b - 1), excess[0]),                 # target at first position
        (0, min(len(B) - 1, b - 1), excess[min(len(B) - 1, b - 1)]),  # target at last position
        (b - 2, min(len(B) - 1, (3 * b) + 1), excess[b + 1]),   # first middle full block
        (b - 2, min(len(B) - 1, (4 * b) + 1), excess[(3 * b)]), # later middle full block
        (b - 2, min(len(B) - 1, (3 * b) + 1), max(excess) + 1), # no result
    ]

    if len(B) > (1 << 15) + 10:
        cases.append(((1 << 15) - 5, (1 << 15) + 5, excess[(1 << 15)]))

    if len(B) > (2 * (1 << 15)) + 10:
        cases.append((len(B) - 12, len(B) - 1, excess[len(B) - 1]))

    for lo, hi, target in cases:
        assert_range_query(B, lo, hi, target)


def test_range_searches_one_bucket():
    for builder in PATTERNS.values():
        assert_range_searches(builder(64))


def test_range_searches_multiple_buckets():
    n = ((3 * (1 << 15)) + 50) // 2
    for builder in PATTERNS.values():
        assert_range_searches(builder(n))


def test_range_searches_final_partial_bucket():
    n = ((2 * (1 << 15)) + 14) // 2
    for builder in PATTERNS.values():
        assert_range_searches(builder(n))
