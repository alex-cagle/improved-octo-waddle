import random

import numpy as np

from bp import BP


BETA = 1 << 15
SEED = 20250217


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


def excess_ref(B, i):
    opens = int(B[: i + 1].sum())
    return (2 * opens) - i - 1


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


def close_ref(B, i):
    if not B[i]:
        return i
    return fwdsearch_ref(B, i, -1)


def open_ref(B, i):
    if B[i] or i <= 0:
        return i
    return bwdsearch_ref(B, i, 0) + 1


def enclose_ref(B, i):
    if B[i]:
        return bwdsearch_ref(B, i, -2) + 1
    return bwdsearch_ref(B, i - 1, -2) + 1


def parent_ref(B, i):
    if i == 0 or i == (len(B) - 1):
        return -1
    return enclose_ref(B, i)


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


def find_close_category_positions(B, beta):
    open_to_close, _ = matching_maps(B)
    last_bucket = (len(B) - 1) // beta
    has_partial = (len(B) % beta) != 0
    categories = {}

    for open_idx, close_idx in open_to_close.items():
        src_bucket = open_idx // beta
        dst_bucket = close_idx // beta

        if "current" not in categories and dst_bucket == src_bucket:
            categories["current"] = open_idx
        if "next" not in categories and dst_bucket == src_bucket + 1:
            categories["next"] = open_idx
        if "several" not in categories and dst_bucket >= src_bucket + 2:
            categories["several"] = open_idx
        if ("final_partial" not in categories and has_partial and
                dst_bucket == last_bucket and src_bucket < dst_bucket):
            categories["final_partial"] = open_idx

    return categories


def find_parent_category_positions(B, beta):
    open_to_parent, close_to_parent = parent_maps(B)
    last_bucket = (len(B) - 1) // beta
    has_partial = (len(B) % beta) != 0
    categories = {"none_root": 0, "none_last": len(B) - 1}

    for pos in range(1, len(B) - 1):
        if B[pos]:
            parent = open_to_parent[pos]
        else:
            parent = close_to_parent[pos]
        src_bucket = pos // beta

        if parent == -1:
            continue

        parent_bucket = parent // beta
        if "current" not in categories and parent_bucket == src_bucket:
            categories["current"] = pos
        if "prev" not in categories and parent_bucket == src_bucket - 1:
            categories["prev"] = pos
        if "several" not in categories and parent_bucket <= src_bucket - 2:
            categories["several"] = pos
        if ("source_final_partial" not in categories and has_partial and
                src_bucket == last_bucket):
            categories["source_final_partial"] = pos

    return categories


TEST_CASES = {
    "nested_slightly_above": nested_bits((BETA + 2) // 2),
    "flat_slightly_above": flat_bits((BETA + 2) // 2),
    "mixed_over_two_buckets": mixed_bits(((2 * BETA) + 14) // 2),
    "nested_over_two_buckets": nested_bits(((2 * BETA) + 14) // 2),
    "random_final_partial": random_bits(((2 * BETA) + 50) // 2, SEED),
}


def test_close_cross_bucket_against_reference():
    seen = set()

    for B in TEST_CASES.values():
        bp = BP(B)
        positions = find_close_category_positions(B, BETA)

        for category, open_idx in positions.items():
            seen.add(category)
            assert bp.close(open_idx) == close_ref(B, open_idx)

    assert {"current", "next", "several", "final_partial"} <= seen


def test_parent_cross_bucket_against_reference():
    seen = set()

    for B in TEST_CASES.values():
        bp = BP(B)
        positions = find_parent_category_positions(B, BETA)

        for category, pos in positions.items():
            seen.add(category)
            assert bp.parent(pos) == parent_ref(B, pos)

    assert {"current", "prev", "several", "source_final_partial",
            "none_root", "none_last"} <= seen
