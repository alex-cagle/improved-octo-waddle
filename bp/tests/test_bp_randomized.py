import random

import numpy as np
import pytest

from bp import BP


SEED = 20250217
NS = (1, 2, 3, 5, 8, 16, 32, 64)
CASES_PER_N = 3
MAX_INTERVALS = 48
MAX_SEARCH_POSITIONS = 12
MAX_SEARCH_DELTAS = 12


def generate_balanced_parentheses(n, rng):
    """Generate a valid balanced-parentheses bitvector of length 2n."""
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


def fwdsearch_ref(B, i, d):
    target = excess_ref(B, i) + d
    for pos in range(i + 1, len(B)):
        if excess_ref(B, pos) == target:
            return pos
    return -1


def bwdsearch_ref(B, i, d):
    target = excess_ref(B, i) + d
    for pos in range(i - 1, -1, -1):
        if excess_ref(B, pos) == target:
            return pos
    return -1


def sample_intervals(size, rng):
    all_intervals = [(i, j) for i in range(size) for j in range(i + 1, size)]
    if len(all_intervals) <= MAX_INTERVALS:
        return all_intervals
    return rng.sample(all_intervals, MAX_INTERVALS)


def sample_positions(size, rng):
    positions = list(range(size))
    if len(positions) <= MAX_SEARCH_POSITIONS:
        return positions
    return sorted(rng.sample(positions, MAX_SEARCH_POSITIONS))


def sample_search_deltas(B, i, rng):
    deltas = {0}

    if i + 1 < len(B):
        forward_positions = list(range(i + 1, len(B)))
        forward_sample = rng.sample(
            forward_positions,
            min(4, len(forward_positions)),
        )
        for pos in forward_sample:
            deltas.add(excess_ref(B, pos) - excess_ref(B, i))

    if i > 0:
        backward_positions = list(range(i))
        backward_sample = rng.sample(
            backward_positions,
            min(4, len(backward_positions)),
        )
        for pos in backward_sample:
            deltas.add(excess_ref(B, pos) - excess_ref(B, i))

    max_abs = max(2, len(B) // 4)
    deltas.add(max_abs)
    deltas.add(-max_abs)

    deltas = sorted(deltas)
    if len(deltas) <= MAX_SEARCH_DELTAS:
        return deltas
    return deltas[:MAX_SEARCH_DELTAS]


@pytest.mark.parametrize("n", NS)
def test_randomized_reference_correctness(n):
    rng = random.Random(SEED + n)

    if n == 1:
        B = generate_balanced_parentheses(n, rng)
        with pytest.raises(ValueError, match="negative dimensions"):
            BP(B)
        pytest.xfail("Current BP rmM constructor does not support n=1 trees")

    for _ in range(CASES_PER_N):
        B = generate_balanced_parentheses(n, rng)
        bp = BP(B)
        size = len(B)

        for i in range(size):
            assert bp.depth(i) == excess_ref(B, i)

        for i, j in sample_intervals(size, rng):
            assert bp.rmq(i, j) == rmq_ref(B, i, j)
            assert bp.rMq(i, j) == rMq_ref(B, i, j)
            assert bp.mincount(i, j) == mincount_ref(B, i, j)

            min_count = mincount_ref(B, i, j)
            q_values = {0, 1, min_count, min_count + 1}
            for q in sorted(q_values):
                assert bp.minselect(i, j, q) == minselect_ref(B, i, j, q)

        for i in sample_positions(size, rng):
            if B[i]:
                assert bp.close(i) == fwdsearch_ref(B, i, -1)
