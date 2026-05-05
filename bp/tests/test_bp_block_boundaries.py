from math import ceil, log

import numpy as np

from bp import BP


CASE_NS = {
    "exact_multiple": 6,   # size=12, b=3
    "just_below": 7,       # size=14, b=3, size % b == b - 1
    "just_above": 8,       # size=16, b=3, size % b == 1
    "final_partial": 9,    # size=18, b=4, nontrivial final partial block
}


def rmm_block_size(size):
    return int(ceil(log(float(size)) * log(log(float(size)))))


def nested_bits(n):
    return np.array(([1] * n) + ([0] * n), dtype=np.uint8)


def flat_bits(n):
    return np.array([1, 0] * n, dtype=np.uint8)


def mixed_bits(n):
    if n == 1:
        return np.array([1, 0], dtype=np.uint8)
    return np.array([1] + ([1, 0] * (n - 1)) + [0], dtype=np.uint8)


PATTERNS = {
    "nested": nested_bits,
    "flat": flat_bits,
    "mixed": mixed_bits,
}


def excess_ref(B, i):
    opens = int(B[: i + 1].sum())
    return (2 * opens) - i - 1


def rmq_ref(B, i, j):
    min_pos = i
    min_val = excess_ref(B, i)
    for pos in range(i + 1, j + 1):
        val = excess_ref(B, pos)
        if val < min_val:
            min_val = val
            min_pos = pos
    return min_pos


def rMq_ref(B, i, j):
    max_pos = i
    max_val = excess_ref(B, i)
    for pos in range(i + 1, j + 1):
        val = excess_ref(B, pos)
        if val > max_val:
            max_val = val
            max_pos = pos
    return max_pos


def mincount_ref(B, i, j):
    values = [excess_ref(B, pos) for pos in range(i, j + 1)]
    min_val = min(values)
    return sum(val == min_val for val in values)


def minselect_ref(B, i, j, q):
    if q <= 0:
        return None

    values = [excess_ref(B, pos) for pos in range(i, j + 1)]
    min_val = min(values)
    count = 0

    for offset, val in enumerate(values):
        if val == min_val:
            count += 1
            if count == q:
                return i + offset

    return None


def close_ref(B, i):
    target = excess_ref(B, i) - 1
    for pos in range(i + 1, len(B)):
        if excess_ref(B, pos) == target:
            return pos
    return -1


def boundary_positions(size, b):
    anchors = {
        b - 1,
        b,
        b + 1,
        (2 * b) - 1,
        2 * b,
        (2 * b) + 1,
        size - 2,
        size - 1,
    }
    return sorted(pos for pos in anchors if 0 <= pos < size)


def boundary_intervals(size, b):
    anchors = boundary_positions(size, b)
    intervals = set()

    for pos in anchors:
        lo = max(0, pos - 1)
        hi = min(size - 1, pos + 1)
        if lo < hi:
            intervals.add((lo, hi))

    for pos in anchors:
        if pos < size - 1:
            intervals.add((pos, size - 1))
        if pos > 0:
            intervals.add((0, pos))

    for left, right in zip(anchors, anchors[1:]):
        if left < right:
            intervals.add((left, right))

    if (b - 1) >= 0 and (b + 1) < size:
        intervals.add((b - 1, b + 1))
    if ((2 * b) - 1) >= 0 and ((2 * b) + 1) < size:
        intervals.add(((2 * b) - 1, (2 * b) + 1))

    return sorted(intervals)


def assert_boundary_case_matches_reference(B):
    bp = BP(B)
    size = len(B)
    b = rmm_block_size(size)
    positions = boundary_positions(size, b)
    intervals = boundary_intervals(size, b)

    for pos in positions:
        assert bp.depth(pos) == excess_ref(B, pos)
        if B[pos]:
            assert bp.close(pos) == close_ref(B, pos)

    for i, j in intervals:
        assert bp.rmq(i, j) == rmq_ref(B, i, j)
        assert bp.rMq(i, j) == rMq_ref(B, i, j)
        assert bp.mincount(i, j) == mincount_ref(B, i, j)

        count = mincount_ref(B, i, j)
        for q in (1, count, count + 1):
            assert bp.minselect(i, j, q) == minselect_ref(B, i, j, q)


def test_block_boundary_cases_against_reference():
    for n in CASE_NS.values():
        for builder in PATTERNS.values():
            assert_boundary_case_matches_reference(builder(n))


def test_leftmost_tie_behavior_across_block_boundaries():
    B = flat_bits(CASE_NS["final_partial"])
    size = len(B)
    b = rmm_block_size(size)

    lo = max(0, b - 1)
    hi = min(size - 1, (2 * b) + 1)

    rmq_exp = rmq_ref(B, lo, hi)
    rMq_exp = rMq_ref(B, lo, hi)

    bp = BP(B)
    assert bp.rmq(lo, hi) == rmq_exp
    assert bp.rMq(lo, hi) == rMq_exp

    # Explicit tie expectations for the flat pattern.
    assert rmq_exp == lo
    if lo + 1 <= hi:
        assert rMq_exp == lo + 1


def test_minselect_q_behavior_across_block_boundaries():
    B = flat_bits(CASE_NS["final_partial"])
    size = len(B)
    b = rmm_block_size(size)
    lo = max(0, b - 1)
    hi = min(size - 1, (2 * b) + 1)

    bp = BP(B)
    count = mincount_ref(B, lo, hi)

    assert bp.minselect(lo, hi, 1) == minselect_ref(B, lo, hi, 1)
    assert bp.minselect(lo, hi, count) == minselect_ref(B, lo, hi, count)
    assert bp.minselect(lo, hi, count + 1) is None
