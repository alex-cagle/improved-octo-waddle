# cython: boundscheck=False, wraparound=False, cdivision=True, linetrace=False
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, BP development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

### NOTE: some doctext strings are copied and pasted from manuscript
### http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf

from libc.math cimport ceil, log as ln, pow, log2
import time
#import numpy.testing as npt
import numpy as np
cimport numpy as np
cimport cython

from bp._binary_tree cimport * #bt_node_from_left, bt_left_child, bt_right_child
from bp._ba cimport *

np.import_array()

cdef extern from "limits.h":
    int INT_MAX


DOUBLE = np.float64
SIZE = np.intp
BOOL = np.uint8
INT32 = np.int32
DEF BUCKET_SIZE = 1 << 15
DEF RMQ_MAX_COVER_NODES = 128
cdef int RMQ_SUBTREE_THRESHOLD = 8


cdef inline int min(int a, int b) nogil:
    if a > b:
        return b
    else:
        return a


cdef inline int max(int a, int b) nogil:
    if a > b:
        return a
    else:
        return b


cdef inline int _excess_from_block_seed(BP self, SIZE_t i) nogil:
    cdef SIZE_t k
    cdef SIZE_t node
    cdef SIZE_t s
    cdef SIZE_t j
    cdef int excess

    k = i // self._rmm.b
    node = self._rmm.n_internal + k
    s = k * self._rmm.b
    excess = (2 * self._rmm.r[node]) - s

    for j in range(s, i + 1):
        excess += -1 + (2 * self._b_ptr[j])

    return excess


cdef inline int _scan_block_forward_rmm(BP self, int i, int k, int d) nogil:
    cdef int lower_bound
    cdef int upper_bound
    cdef int node
    cdef int s
    cdef int j
    cdef int excess

    s = k * self._rmm.b
    lower_bound = max(i + 1, s)
    upper_bound = min((k + 1) * self._rmm.b, self.size)

    node = self._rmm.n_internal + k
    excess = (2 * self._rmm.r[node]) - s

    for j in range(s, lower_bound):
        excess += -1 + (2 * self._b_ptr[j])

    for j in range(lower_bound, upper_bound):
        excess += -1 + (2 * self._b_ptr[j])
        if excess == d:
            return j

    return -1


cdef inline int _scan_block_backward_rmm(BP self, int i, int k, int d) nogil:
    cdef int lower_bound
    cdef int upper_bound
    cdef int node
    cdef int s
    cdef int j
    cdef int excess

    s = k * self._rmm.b
    lower_bound = s - 1
    if lower_bound >= 0:
        lower_bound -= 1

    upper_bound = min((k + 1) * self._rmm.b, self.size) - 1
    upper_bound = min(i - 1, upper_bound)

    if upper_bound <= 0:
        return -1

    node = self._rmm.n_internal + k
    excess = (2 * self._rmm.r[node]) - s

    for j in range(s, upper_bound + 1):
        excess += -1 + (2 * self._b_ptr[j])

    for j in range(upper_bound, lower_bound, -1):
        if excess == d:
            return j
        excess -= -1 + (2 * self._b_ptr[j])

    return -1


cdef int _bucket_find_first_containing_node(BP self, int node, int node_lo,
                                            int node_hi, int lo_bucket,
                                            int hi_bucket, int target) nogil:
    cdef int mid
    cdef int result

    if node_hi < lo_bucket or node_lo > hi_bucket:
        return -1

    if self.bucket_tree_m[node] > target or self.bucket_tree_M[node] < target:
        return -1

    if node_lo == node_hi:
        if node_lo >= self.n_buckets:
            return -1
        return node_lo

    mid = node_lo + ((node_hi - node_lo) // 2)
    result = _bucket_find_first_containing_node(
        self, 2 * node, node_lo, mid, lo_bucket, hi_bucket, target
    )
    if result != -1:
        return result

    return _bucket_find_first_containing_node(
        self, (2 * node) + 1, mid + 1, node_hi, lo_bucket, hi_bucket, target
    )


cdef int _bucket_find_last_containing_node(BP self, int node, int node_lo,
                                           int node_hi, int lo_bucket,
                                           int hi_bucket, int target) nogil:
    cdef int mid
    cdef int result

    if node_hi < lo_bucket or node_lo > hi_bucket:
        return -1

    if self.bucket_tree_m[node] > target or self.bucket_tree_M[node] < target:
        return -1

    if node_lo == node_hi:
        if node_lo >= self.n_buckets:
            return -1
        return node_lo

    mid = node_lo + ((node_hi - node_lo) // 2)
    result = _bucket_find_last_containing_node(
        self, (2 * node) + 1, mid + 1, node_hi, lo_bucket, hi_bucket, target
    )
    if result != -1:
        return result

    return _bucket_find_last_containing_node(
        self, 2 * node, node_lo, mid, lo_bucket, hi_bucket, target
    )


cdef int _bucket_find_first_containing(BP self, int lo_bucket, int hi_bucket,
                                       int target) nogil:
    if hi_bucket < 0 or lo_bucket >= self.n_buckets:
        return -1

    lo_bucket = max(0, lo_bucket)
    hi_bucket = min(self.n_buckets - 1, hi_bucket)
    if lo_bucket > hi_bucket:
        return -1

    return _bucket_find_first_containing_node(
        self, 1, 0, self.bucket_tree_base - 1, lo_bucket, hi_bucket, target
    )


cdef int _bucket_find_last_containing(BP self, int lo_bucket, int hi_bucket,
                                      int target) nogil:
    if hi_bucket < 0 or lo_bucket >= self.n_buckets:
        return -1

    lo_bucket = max(0, lo_bucket)
    hi_bucket = min(self.n_buckets - 1, hi_bucket)
    if lo_bucket > hi_bucket:
        return -1

    return _bucket_find_last_containing_node(
        self, 1, 0, self.bucket_tree_base - 1, lo_bucket, hi_bucket, target
    )


cdef int _rmm_find_first_cover_node_containing(BP self, int lo_block,
                                               int hi_block,
                                               int target) nogil:
    cdef SIZE_t left_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t right_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t left_count = 0
    cdef SIZE_t right_count = 0
    cdef SIZE_t base
    cdef SIZE_t l
    cdef SIZE_t r
    cdef SIZE_t node
    cdef SIZE_t idx

    if lo_block > hi_block:
        return -1

    base = self._rmm.n_internal + 1
    l = base + lo_block
    r = base + hi_block

    while l <= r:
        if l % 2 == 1:
            left_nodes[left_count] = l - 1
            left_count += 1
            l += 1

        if r % 2 == 0:
            right_nodes[right_count] = r - 1
            right_count += 1
            r -= 1

        l //= 2
        r //= 2

    for idx in range(left_count):
        node = left_nodes[idx]
        if self._rmm.mM[node, self._rmm.m_idx] <= target <= self._rmm.mM[node, self._rmm.M_idx]:
            return node

    while right_count > 0:
        right_count -= 1
        node = right_nodes[right_count]
        if self._rmm.mM[node, self._rmm.m_idx] <= target <= self._rmm.mM[node, self._rmm.M_idx]:
            return node

    return -1


cdef int _rmm_find_last_cover_node_containing(BP self, int lo_block,
                                              int hi_block,
                                              int target) nogil:
    cdef SIZE_t left_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t right_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t left_count = 0
    cdef SIZE_t right_count = 0
    cdef SIZE_t base
    cdef SIZE_t l
    cdef SIZE_t r
    cdef SIZE_t node
    cdef SIZE_t idx

    if lo_block > hi_block:
        return -1

    base = self._rmm.n_internal + 1
    l = base + lo_block
    r = base + hi_block

    while l <= r:
        if l % 2 == 1:
            left_nodes[left_count] = l - 1
            left_count += 1
            l += 1

        if r % 2 == 0:
            right_nodes[right_count] = r - 1
            right_count += 1
            r -= 1

        l //= 2
        r //= 2

    for idx in range(right_count):
        node = right_nodes[idx]
        if self._rmm.mM[node, self._rmm.m_idx] <= target <= self._rmm.mM[node, self._rmm.M_idx]:
            return node

    while left_count > 0:
        left_count -= 1
        node = left_nodes[left_count]
        if self._rmm.mM[node, self._rmm.m_idx] <= target <= self._rmm.mM[node, self._rmm.M_idx]:
            return node

    return -1


cdef inline SIZE_t _rmm_descend_leftmost_leaf_containing(BP self, SIZE_t node,
                                                         int target) nogil:
    cdef SIZE_t left_child
    cdef SIZE_t right_child

    while not bt_is_leaf(node, self._rmm.height):
        left_child = bt_left_child(node)
        right_child = bt_right_child(node)
        if self._rmm.mM[left_child, self._rmm.m_idx] <= target <= self._rmm.mM[left_child, self._rmm.M_idx]:
            node = left_child
        else:
            node = right_child

    return node


cdef inline SIZE_t _rmm_descend_rightmost_leaf_containing(BP self, SIZE_t node,
                                                          int target) nogil:
    cdef SIZE_t left_child
    cdef SIZE_t right_child

    while not bt_is_leaf(node, self._rmm.height):
        left_child = bt_left_child(node)
        right_child = bt_right_child(node)
        if self._rmm.mM[right_child, self._rmm.m_idx] <= target <= self._rmm.mM[right_child, self._rmm.M_idx]:
            node = right_child
        else:
            node = left_child

    return node


cdef int _fwdsearch_in_range(BP self, SIZE_t lo, SIZE_t hi, int target) nogil:
    cdef int first_block
    cdef int last_block
    cdef int first_block_end
    cdef int last_block_start
    cdef int k
    cdef int result
    cdef SIZE_t node
    cdef SIZE_t leaf

    if lo > hi or hi >= self.size:
        return -1

    first_block = lo // self._rmm.b
    last_block = hi // self._rmm.b

    if first_block == last_block:
        result = _scan_block_forward_rmm(self, <int>lo - 1, first_block, target)
        if result != -1 and result <= hi:
            return result
        return -1

    first_block_end = min((first_block + 1) * self._rmm.b, self.size) - 1
    result = _scan_block_forward_rmm(self, <int>lo - 1, first_block, target)
    if result != -1 and result <= first_block_end:
        return result

    if first_block + 1 <= last_block - 1:
        node = _rmm_find_first_cover_node_containing(
            self, first_block + 1, last_block - 1, target
        )
        if node != <SIZE_t>-1:
            leaf = _rmm_descend_leftmost_leaf_containing(self, node, target)
            k = leaf - self._rmm.n_internal
            result = _scan_block_forward_rmm(self, (k * self._rmm.b) - 1, k, target)
            if result != -1:
                return result

    last_block_start = last_block * self._rmm.b
    result = _scan_block_forward_rmm(self, last_block_start - 1, last_block, target)
    if result != -1 and result <= hi:
        return result

    return -1


cdef int _bwdsearch_in_range(BP self, SIZE_t lo, SIZE_t hi, int target) nogil:
    cdef int first_block
    cdef int last_block
    cdef int first_block_end
    cdef int last_block_start
    cdef int block_end
    cdef int k
    cdef int result
    cdef SIZE_t node
    cdef SIZE_t leaf

    if lo > hi or hi >= self.size:
        return -1

    if lo == 0 and hi == 0:
        if _excess_from_block_seed(self, 0) == target:
            return 0
        return -1

    first_block = lo // self._rmm.b
    last_block = hi // self._rmm.b

    if first_block == last_block:
        result = _scan_block_backward_rmm(self, <int>hi + 1, first_block, target)
        if result != -1 and result >= lo:
            return result
        return -1

    last_block_start = last_block * self._rmm.b
    result = _scan_block_backward_rmm(self, <int>hi + 1, last_block, target)
    if result != -1 and result >= last_block_start:
        return result

    if first_block + 1 <= last_block - 1:
        node = _rmm_find_last_cover_node_containing(
            self, first_block + 1, last_block - 1, target
        )
        if node != <SIZE_t>-1:
            leaf = _rmm_descend_rightmost_leaf_containing(self, node, target)
            k = leaf - self._rmm.n_internal
            block_end = min((k + 1) * self._rmm.b, self.size)
            result = _scan_block_backward_rmm(self, block_end, k, target)
            if result != -1:
                return result

    first_block_end = min((first_block + 1) * self._rmm.b, self.size) - 1
    result = _scan_block_backward_rmm(self, first_block_end + 1, first_block, target)
    if result != -1 and result >= lo:
        return result

    return -1


cdef inline SIZE_t _rmq_scan_range(BP self, SIZE_t lo, SIZE_t hi, int* min_v) nogil:
    cdef SIZE_t pos
    cdef SIZE_t min_k
    cdef int excess

    min_k = lo
    excess = _excess_from_block_seed(self, lo)
    min_v[0] = excess

    for pos in range(lo + 1, hi + 1):
        excess += -1 + (2 * self._b_ptr[pos])
        if excess < min_v[0]:
            min_v[0] = excess
            min_k = pos

    return min_k


cdef inline SIZE_t _rMq_scan_range(BP self, SIZE_t lo, SIZE_t hi, int* max_v) nogil:
    cdef SIZE_t pos
    cdef SIZE_t max_k
    cdef int excess

    max_k = lo
    excess = _excess_from_block_seed(self, lo)
    max_v[0] = excess

    for pos in range(lo + 1, hi + 1):
        excess += -1 + (2 * self._b_ptr[pos])
        if excess > max_v[0]:
            max_v[0] = excess
            max_k = pos

    return max_k


def _test_bucket_find_first_containing(BP self, int lo_bucket, int hi_bucket,
                                       int target):
    return _bucket_find_first_containing(self, lo_bucket, hi_bucket, target)


def _test_bucket_find_last_containing(BP self, int lo_bucket, int hi_bucket,
                                      int target):
    return _bucket_find_last_containing(self, lo_bucket, hi_bucket, target)


def _test_fwdsearch_in_range(BP self, SIZE_t lo, SIZE_t hi, int target):
    return _fwdsearch_in_range(self, lo, hi, target)


def _test_bwdsearch_in_range(BP self, SIZE_t lo, SIZE_t hi, int target):
    return _bwdsearch_in_range(self, lo, hi, target)


cdef void _rmq_find_best_middle_cover_node(BP self, SIZE_t query_left,
                                           SIZE_t query_right,
                                           SIZE_t* best_node,
                                           int* best_min) nogil:
    # Each side of the canonical cover contributes at most one node per rmM
    # tree level, so a fixed per-side bound is sufficient here.
    cdef SIZE_t left_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t right_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t left_count = 0
    cdef SIZE_t right_count = 0
    cdef SIZE_t base
    cdef SIZE_t l
    cdef SIZE_t r
    cdef SIZE_t node
    cdef SIZE_t idx
    cdef int node_min

    base = self._rmm.n_internal + 1
    l = base + query_left
    r = base + query_right

    while l <= r:
        if l % 2 == 1:
            left_nodes[left_count] = l - 1
            left_count += 1
            l += 1

        if r % 2 == 0:
            right_nodes[right_count] = r - 1
            right_count += 1
            r -= 1

        l //= 2
        r //= 2

    for idx in range(left_count):
        node = left_nodes[idx]
        node_min = <int>self._rmm.mM[node, self._rmm.m_idx]
        if node_min < best_min[0]:
            best_min[0] = node_min
            best_node[0] = node

    while right_count > 0:
        right_count -= 1
        node = right_nodes[right_count]
        node_min = <int>self._rmm.mM[node, self._rmm.m_idx]
        if node_min < best_min[0]:
            best_min[0] = node_min
            best_node[0] = node


cdef inline SIZE_t _rmq_descend_leftmost_leaf(BP self, SIZE_t node,
                                              int target_min) nogil:
    cdef SIZE_t left_child

    while not bt_is_leaf(node, self._rmm.height):
        left_child = bt_left_child(node)
        if self._rmm.mM[left_child, self._rmm.m_idx] == target_min:
            node = left_child
        else:
            node = bt_right_child(node)

    return node


cdef void _rMq_find_best_middle_cover_node(BP self, SIZE_t query_left,
                                           SIZE_t query_right,
                                           SIZE_t* best_node,
                                           int* best_max) nogil:
    # Each side of the canonical cover contributes at most one node per rmM
    # tree level, so a fixed per-side bound is sufficient here.
    cdef SIZE_t left_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t right_nodes[RMQ_MAX_COVER_NODES]
    cdef SIZE_t left_count = 0
    cdef SIZE_t right_count = 0
    cdef SIZE_t base
    cdef SIZE_t l
    cdef SIZE_t r
    cdef SIZE_t node
    cdef SIZE_t idx
    cdef int node_max

    base = self._rmm.n_internal + 1
    l = base + query_left
    r = base + query_right

    while l <= r:
        if l % 2 == 1:
            left_nodes[left_count] = l - 1
            left_count += 1
            l += 1

        if r % 2 == 0:
            right_nodes[right_count] = r - 1
            right_count += 1
            r -= 1

        l //= 2
        r //= 2

    for idx in range(left_count):
        node = left_nodes[idx]
        node_max = <int>self._rmm.mM[node, self._rmm.M_idx]
        if node_max > best_max[0]:
            best_max[0] = node_max
            best_node[0] = node

    while right_count > 0:
        right_count -= 1
        node = right_nodes[right_count]
        node_max = <int>self._rmm.mM[node, self._rmm.M_idx]
        if node_max > best_max[0]:
            best_max[0] = node_max
            best_node[0] = node


cdef inline SIZE_t _rMq_descend_leftmost_leaf(BP self, SIZE_t node,
                                              int target_max) nogil:
    cdef SIZE_t left_child

    while not bt_is_leaf(node, self._rmm.height):
        left_child = bt_left_child(node)
        if self._rmm.mM[left_child, self._rmm.M_idx] == target_max:
            node = left_child
        else:
            node = bt_right_child(node)

    return node


cdef class mM:
    def __cinit__(self, BOOL_t[:] B, int B_size):
        self.m_idx = 0
        self.M_idx = 1

        self.rmm(B, B_size)

    cdef void rmm(self, BOOL_t[:] B, int B_size) nogil:
        """Construct the rmM tree based off of Navarro and Sadakane

        http://www.dcc.uchile.cl/~gnavarro/ps/talg12.pdf
        """
        cdef int i, j, lvl, pos  # for loop support
        cdef int offset  # tip offset in binary tree for a given parenthesis
        cdef int lower_limit  # the lower limit of the bucket a parenthesis is in
        cdef int upper_limit  # the upper limit of the bucket a parenthesis is in
        cdef int min_ = 0 # m, absolute minimum for a blokc
        cdef int max_ = 0 # M, absolute maximum for a block
        cdef int excess = 0 # e, absolute excess
        cdef int vbar
        cdef int r = 0

        # build tip info
        self.b = max(1, <int>ceil(ln(<double> B_size) * ln(ln(<double> B_size))))

        # determine the number of nodes and height of the binary tree
        self.n_tip = <int>ceil(B_size / <double> self.b)
        self.height = <int>ceil(log2(self.n_tip))
        self.n_internal = <int>(pow(2, self.height)) - 1
        self.n_total = self.n_tip + self.n_internal

        with gil:
            # creation of a memoryview directly or via numpy requires the GIL:
            # http://stackoverflow.com/a/22238012
            self.mM = np.zeros((self.n_total, 2), dtype=SIZE)
            self.r = np.zeros(self.n_total, dtype=SIZE)

        # annoying, cannot do step in range if step is not known at runtime
        # see https://github.com/cython/cython/pull/520
        # for i in range(0, B_size, b):
        # as a result, doing a custom range using a while loop
        # compute for tips of rmM tree
        i = 0
        while i < B_size:
            offset = i // self.b
            lower_limit = i
            upper_limit = min(i + self.b, B_size)
            min_ = INT_MAX
            max_ = 0
            
            self.r[offset + self.n_internal] = r 
            for j in range(lower_limit, upper_limit):
                # G function, a +-1 method where if B[j] == 1 we +1, and if
                # B[j] == 0 we -1
                excess += -1 + (2 * B[j]) 
                r += B[j]

                if excess < min_:
                    min_ = excess

                if excess > max_:
                    max_ = excess

                # at the left bound of the bucket
            
            self.mM[offset + self.n_internal, self.m_idx] = min_
            self.mM[offset + self.n_internal, self.M_idx] = max_

            i += self.b

        # compute for internal nodes of rmM tree in reverse level order starting 
        # at the level above the tips
        for lvl in range(self.height - 1, -1, -1):
            num_curr_nodes = <int>pow(2, lvl)

            # for each node in the level
            for pos in range(num_curr_nodes):
                # obtain the node, and the index to its children
                node = bt_node_from_left(pos, lvl)
                lchild = bt_left_child(node)
                rchild = bt_right_child(node)
                
                if lchild >= self.n_total:
                    continue

                elif rchild >= self.n_total:
                    self.mM[node, self.m_idx] = self.mM[lchild, self.m_idx] 
                    self.mM[node, self.M_idx] = self.mM[lchild, self.M_idx]
                else:    
                    self.mM[node, self.m_idx] = min(self.mM[lchild, self.m_idx], 
                                                    self.mM[rchild, self.m_idx])
                    self.mM[node, self.M_idx] = max(self.mM[lchild, self.M_idx], 
                                                    self.mM[rchild, self.M_idx])

                self.r[node] = self.r[lchild] 
                    

@cython.final
cdef class BP:
    """A balanced parentheses succinct data structure tree representation

    The basis for this implementation is the data structure described by
    Cordova and Navarro [1]. In some instances, some docstring text was copied
    verbatim from the manuscript. This does not implement the bucket-based
    trees, although that would be a very interesting next step. 

    A node in this data structure is represented by 2 bits, an open parenthesis
    and a close parenthesis. The implementation uses a numpy uint8 type where
    an open parenthesis is a 1 and a close is a 0. In general, operations on
    this tree are best suited for passing in the opening parenthesis index, so
    for instance, if you'd like to use BP.isleaf to determine if a node is a 
    leaf, the operation is defined only for using the opening parenthesis. At 
    this time, there is some ambiguity over what methods can handle a closing
    parenthesis.

    Node attributes, such as names, are stored external to this data structure.

    The motivator for this data structure is pure performance both in space and
    time. As such, there is minimal sanity checking. It is advised to use this
    structure with care, and ideally within a framework which can assure 
    sanity. 

    References
    ----------
    [1] http://www.dcc.uchile.cl/~gnavarro/ps/tcs16.2.pdf
    """

    def __cinit__(self, np.ndarray[BOOL_t, ndim=1] B, 
                  np.ndarray[DOUBLE_t, ndim=1] lengths=None,
                  np.ndarray[object, ndim=1] names=None,
                  np.ndarray[INT32_t, ndim=1] edges=None):
        cdef SIZE_t i
        cdef SIZE_t size
        cdef SIZE_t j
        cdef SIZE_t bucket_idx
        cdef SIZE_t bucket_start
        cdef SIZE_t bucket_end
        cdef SIZE_t[:] _k_index_0
        cdef SIZE_t[:] _k_index_1
        cdef SIZE_t[:] _r_index_0
        cdef SIZE_t[:] _r_index_1
        cdef SIZE_t[:] bucket_e
        cdef SIZE_t[:] bucket_m
        cdef SIZE_t[:] bucket_M
        cdef SIZE_t[:] bucket_tree_m
        cdef SIZE_t[:] bucket_tree_M
        cdef np.ndarray[object, ndim=1] _names
        cdef np.ndarray[DOUBLE_t, ndim=1] _lengths
        cdef np.ndarray[INT32_t, ndim=1] _edges
        cdef np.ndarray[SIZE_t, ndim=1] _edge_lookup
        cdef int excess
        cdef int bucket_min
        cdef int bucket_max
        cdef int node
        cdef int left
        cdef int right

        # the tree is only valid if it is balanaced
        assert B.sum() == (float(B.size) / 2)
        self.B = B
        self._b_ptr = &B[0]
        self.size = B.size
        self.beta = BUCKET_SIZE
        self.n_buckets = max(1, <int>ceil(self.size / <double> self.beta))
        self.bucket_tree_base = 1
        while self.bucket_tree_base < self.n_buckets:
            self.bucket_tree_base *= 2

        bucket_e = np.zeros(self.n_buckets, dtype=SIZE)
        bucket_m = np.zeros(self.n_buckets, dtype=SIZE)
        bucket_M = np.zeros(self.n_buckets, dtype=SIZE)
        bucket_tree_m = np.full(2 * self.bucket_tree_base, INT_MAX, dtype=SIZE)
        bucket_tree_M = np.zeros(2 * self.bucket_tree_base, dtype=SIZE)

        excess = 0
        for bucket_idx in range(self.n_buckets):
            bucket_start = bucket_idx * self.beta
            bucket_end = min((bucket_idx + 1) * self.beta, self.size) - 1
            bucket_min = INT_MAX
            bucket_max = 0

            for j in range(bucket_start, bucket_end + 1):
                excess += -1 + (2 * self._b_ptr[j])
                if excess < bucket_min:
                    bucket_min = excess
                if excess > bucket_max:
                    bucket_max = excess

            bucket_e[bucket_idx] = excess
            bucket_m[bucket_idx] = bucket_min
            bucket_M[bucket_idx] = bucket_max
            bucket_tree_m[self.bucket_tree_base + bucket_idx] = bucket_min
            bucket_tree_M[self.bucket_tree_base + bucket_idx] = bucket_max

        for node in range(self.bucket_tree_base - 1, 0, -1):
            left = 2 * node
            right = left + 1
            bucket_tree_m[node] = min(bucket_tree_m[left], bucket_tree_m[right])
            bucket_tree_M[node] = max(bucket_tree_M[left], bucket_tree_M[right])

        self.bucket_e = bucket_e
        self.bucket_m = bucket_m
        self.bucket_M = bucket_M
        self.bucket_tree_m = bucket_tree_m
        self.bucket_tree_M = bucket_tree_M

        self._rmm = mM(B, B.size)

        if names is not None:
            self._names = names
        else:
            self._names = np.full(self.B.size, None, dtype=object)

        if lengths is not None:
            self._lengths = lengths
        else:
            self._lengths = np.zeros(self.B.size, dtype=DOUBLE)

        if edges is not None:
            self._set_edges(edges)
        else:
            self._edges = np.full(self.B.size, 0, dtype=INT32)
            self._edge_lookup = None

        # precursor for select index cache
        _r_index_0 = np.cumsum((1 - B), dtype=SIZE)
        _r_index_1 = np.cumsum(B, dtype=SIZE)

        # construct a select index. These operations are performed frequently,
        # and easy to cache at a relatively minor memory expense. It cannot be
        # assumed that open and close will be same length so can't stack
        #TODO: leverage rmmtree, and calculate select on the fly
        _k_index_0 = np.unique(_r_index_0,
                               return_index=True)[1].astype(SIZE)
        self._k_index_0 = _k_index_0
        _k_index_1 = np.unique(_r_index_1,
                               return_index=True)[1].astype(SIZE)
        self._k_index_1 = _k_index_1

    def write(self, object fname):
        np.savez_compressed(fname, names=self._names, lengths=self._lengths, 
                            B=self.B)

    @staticmethod 
    def read(object fname):
        data = np.load(fname)
        bp = BP(data['B'], names=data['names'], lengths=data['lengths'])
        return bp

    def set_names(self, np.ndarray[object, ndim=1] names):
        self._names = names

    def set_lengths(self, np.ndarray[DOUBLE_t, ndim=1] lengths):
        self._lengths = lengths
   
    cdef void _set_edges(self, np.ndarray[INT32_t, ndim=1] edges):
        cdef:
            int i, n
            INT32_t edge
            np.ndarray[SIZE_t, ndim=1] _edge_lookup
            np.ndarray[BOOL_t, ndim=1] b

        b = self.B
        n = b.size
       
        _edge_lookup = np.full(n, 0, dtype=SIZE)
        for i in range(n):
            if b[i] == 1:
                edge = edges[i]
                _edge_lookup[edge] = i
        
        self._edge_lookup = _edge_lookup
        self._edges = edges

    def set_edges(self, np.ndarray[INT32_t, ndim=1] edges):
        self._set_edges(edges)

    cpdef inline unicode name(self, SIZE_t i):
        return self._names[i]

    cpdef inline DOUBLE_t length(self, SIZE_t i):
        return self._lengths[i]

    cpdef inline INT32_t edge(self, SIZE_t i):
        return self._edges[i]

    cpdef SIZE_t edge_from_number(self, INT32_t n):
        return self._edge_lookup[n]

    cdef inline SIZE_t rank(self, SIZE_t t, SIZE_t i) nogil:
        """Determine the rank order of the ith bit t

        Rank is the order of the ith bit observed, from left to right. For
        t=1, this is a preorder traversal of the tree.

        Parameters
        ----------
        t : SIZE_t
            The bit value, either 0 or 1 where 0 is a closing parenthesis and
            1 is an opening.
        i : SIZE_T
            The position to evaluate

        Returns
        -------
        SIZE_t
            The rank order of the position.
        """
        cdef int k
        cdef int r = 0
        cdef int lower_bound
        cdef int upper_bound
        cdef int j
        cdef int node

        #TODO: add method to mM for determining block from i
        k = i // self._rmm.b  
        
        lower_bound = k * self._rmm.b

        # upper_bound is block boundary or end of tree
        upper_bound = min((k + 1) * self._rmm.b, self.size)
        upper_bound = min(upper_bound, i + 1)

        # collect rank from within the block
        for j in range(lower_bound, upper_bound):
            r += self._b_ptr[j]
        
        # collect the rank at the left end of the block
        node = bt_node_from_left(k, self._rmm.height)
        r += self._rmm.r[node]

        # TODO: can this if statement be removed?
        if t:
            return r
        else:
            return (i - r) + 1
 
    cdef inline SIZE_t select(self, SIZE_t t, SIZE_t k) nogil:
        """The position in B of the kth occurrence of the bit t."""
        if t:
            return self._k_index_1[k]
        else:
            return self._k_index_0[k]
        
    cdef SIZE_t _excess(self, SIZE_t i) nogil:
        """Actually compute excess"""
        if i < 0:
            return 0  # wasn't stated as needed but appears so given testing
        return (2 * self.rank(1, i) - i) - 1

    cdef SIZE_t excess(self, SIZE_t i) nogil:
        """the number of opening minus closing parentheses in B[1, i]"""
        # same as: self.rank(1, i) - self.rank(0, i)
        return _excess_from_block_seed(self, i)
    
    cpdef inline SIZE_t close(self, SIZE_t i) nogil:
        """The position of the closing parenthesis that matches B[i]"""
        if not self._b_ptr[i]:
            # identity: the close of a closed parenthesis is itself
            return i

        return self.fwdsearch(i, -1)

    cdef inline SIZE_t open(self, SIZE_t i) nogil:
        """The position of the opening parenthesis that matches B[i]"""
        if self._b_ptr[i] or i <= 0:
            # identity: the open of an open parenthesis is itself
            # the open of 0 is open. A negative index cannot be open, so just return
            return i

        return self.bwdsearch(i, 0) + 1

    cdef inline SIZE_t enclose(self, SIZE_t i) nogil:
        """The opening parenthesis of the smallest matching pair that contains position i"""
        if self._b_ptr[i]:
            return self.bwdsearch(i, -2) + 1
        else:
            return self.bwdsearch(i - 1, -2) + 1

    cpdef SIZE_t rmq(self, SIZE_t i, SIZE_t j) nogil:
        """The leftmost minimum excess in i -> j"""
        cdef:
            SIZE_t k, min_k, obs_k
            SIZE_t first_block, last_block
            SIZE_t first_block_end, last_block_start, block_end
            SIZE_t leaf
            SIZE_t best_node
            SIZE_t mid_left, mid_right
            SIZE_t middle_blocks
            int min_v, obs_v

        first_block = i // self._rmm.b
        last_block = j // self._rmm.b

        if first_block == last_block:
            return _rmq_scan_range(self, i, j, &min_v)

        first_block_end = min((first_block + 1) * self._rmm.b, self.size) - 1
        min_k = _rmq_scan_range(self, i, first_block_end, &min_v)

        if first_block + 1 <= last_block - 1:
            mid_left = first_block + 1
            mid_right = last_block - 1
            middle_blocks = mid_right - mid_left + 1

            if middle_blocks < RMQ_SUBTREE_THRESHOLD:
                for k in range(mid_left, mid_right + 1):
                    leaf = self._rmm.n_internal + k
                    obs_v = <int>self._rmm.mM[leaf, self._rmm.m_idx]
                    if obs_v < min_v:
                        block_end = min((k + 1) * self._rmm.b, self.size) - 1
                        obs_k = _rmq_scan_range(self, k * self._rmm.b, block_end,
                                                &obs_v)
                        min_k = obs_k
                        min_v = obs_v
            else:
                best_node = -1
                obs_v = min_v

                _rmq_find_best_middle_cover_node(self, mid_left, mid_right,
                                                 &best_node, &obs_v)

                if best_node != -1:
                    leaf = _rmq_descend_leftmost_leaf(self, best_node, obs_v)
                    k = leaf - self._rmm.n_internal
                    block_end = min((k + 1) * self._rmm.b, self.size) - 1
                    obs_k = _rmq_scan_range(self, k * self._rmm.b, block_end, &obs_v)
                    min_k = obs_k
                    min_v = obs_v

        last_block_start = last_block * self._rmm.b
        obs_k = _rmq_scan_range(self, last_block_start, j, &obs_v)
        if obs_v < min_v:
            min_k = obs_k
            min_v = obs_v

        return min_k

    cpdef SIZE_t rMq(self, SIZE_t i, SIZE_t j) nogil:
        """The leftmost maximmum excess in i -> j"""
        cdef:
            SIZE_t k, max_k, obs_k
            SIZE_t first_block, last_block
            SIZE_t first_block_end, last_block_start, block_end
            SIZE_t leaf
            SIZE_t best_node
            SIZE_t mid_left, mid_right
            SIZE_t middle_blocks
            int max_v, obs_v

        first_block = i // self._rmm.b
        last_block = j // self._rmm.b

        if first_block == last_block:
            return _rMq_scan_range(self, i, j, &max_v)

        first_block_end = min((first_block + 1) * self._rmm.b, self.size) - 1
        max_k = _rMq_scan_range(self, i, first_block_end, &max_v)

        if first_block + 1 <= last_block - 1:
            mid_left = first_block + 1
            mid_right = last_block - 1
            middle_blocks = mid_right - mid_left + 1

            if middle_blocks < RMQ_SUBTREE_THRESHOLD:
                for k in range(mid_left, mid_right + 1):
                    leaf = self._rmm.n_internal + k
                    obs_v = <int>self._rmm.mM[leaf, self._rmm.M_idx]
                    if obs_v > max_v:
                        block_end = min((k + 1) * self._rmm.b, self.size) - 1
                        obs_k = _rMq_scan_range(self, k * self._rmm.b, block_end,
                                                &obs_v)
                        max_k = obs_k
                        max_v = obs_v
            else:
                best_node = -1
                obs_v = max_v

                _rMq_find_best_middle_cover_node(self, mid_left, mid_right,
                                                 &best_node, &obs_v)

                if best_node != -1:
                    leaf = _rMq_descend_leftmost_leaf(self, best_node, obs_v)
                    k = leaf - self._rmm.n_internal
                    block_end = min((k + 1) * self._rmm.b, self.size) - 1
                    obs_k = _rMq_scan_range(self, k * self._rmm.b, block_end, &obs_v)
                    max_k = obs_k
                    max_v = obs_v

        last_block_start = last_block * self._rmm.b
        obs_k = _rMq_scan_range(self, last_block_start, j, &obs_v)
        if obs_v > max_v:
            max_k = obs_k
            max_v = obs_v

        return max_k

    def __len__(self):
        """The number of nodes in the tree"""
        return self.size / 2

    def __repr__(self):
        """Returns summary of the tree
        
        Returns
        -------
        str
            A summary of this node and all descendants
        
        Notes
        -----
        This method returns the name of the node and a count of tips and the
        number of internal nodes in the tree. This docstring and repr was
        adapted from skbio.TreeNode
        """
        cdef total_nodes = len(self)
        cdef tip_count = self.ntips()

        return "<BP, name: %s, internal node count: %d, tips count: %d>" % \
                (self.name(0), total_nodes - tip_count, tip_count)

    def __reduce__(self):
        return (BP, (self.B, self._lengths, self._names))

    cpdef SIZE_t depth(self, SIZE_t i) nogil:
        """The depth of node i"""
        return self.excess(i)

    cpdef SIZE_t root(self) nogil:
        """The root of the tree"""
        return 0

    cpdef SIZE_t parent(self, SIZE_t i) nogil:
        """The parent of node i"""
        # TODO: only make operations like this defined on the open parentheses.
        # this monkeying with checking open/close sucks.
        if i == self.root() or i == (self.size - 1):
            return -1
        else:
            return self.enclose(i)

    cpdef BOOL_t isleaf(self, SIZE_t i) nogil:
        """Whether the node is a leaf"""
        return self._b_ptr[i] and (not self._b_ptr[i + 1])

    cpdef SIZE_t fchild(self, SIZE_t i) nogil:
        """The first child of i (i.e., the left child)

        fchild(i) = i + 1 (if i is not a leaf)

        Returns 0 if the node is a leaf as the root cannot be a child by
        definition.
        """
        if self._b_ptr[i]:
            if self.isleaf(i):
                return 0
            else:
                return i + 1
        else:
            return self.fchild(self.open(i))

    cpdef SIZE_t lchild(self, SIZE_t i) nogil:
        """The last child of i (i.e., the right child)

        lchild(i) = open(close(i) − 1) (if i is not a leaf)

        Returns 0 if the node is a leaf as the root cannot be a child by
        definition.
        """
        if self._b_ptr[i]:
            if self.isleaf(i):
                return 0
            else:
                return self.open(self.close(i) - 1)
        else:
            return self.lchild(self.open(i))

    def mincount(self, SIZE_t i, SIZE_t j):
        """number of occurrences of the minimum in excess(i), excess(i + 1), . . . , excess(j)."""
        cdef SIZE_t pos
        cdef SIZE_t min_pos
        cdef SIZE_t count
        cdef int min_v
        cdef int excess

        min_pos = self.rmq(i, j)
        min_v = _excess_from_block_seed(self, min_pos)
        excess = _excess_from_block_seed(self, i)
        count = 0

        if excess == min_v:
            count += 1

        for pos in range(i + 1, j + 1):
            excess += -1 + (2 * self._b_ptr[pos])
            if excess == min_v:
                count += 1

        return count

    def minselect(self, SIZE_t i, SIZE_t j, SIZE_t q):
        """position of the qth minimum in excess(i), excess(i + 1), . . . , excess(j)."""
        cdef SIZE_t pos
        cdef SIZE_t min_pos
        cdef SIZE_t count
        cdef int min_v
        cdef int excess

        min_pos = self.rmq(i, j)
        min_v = _excess_from_block_seed(self, min_pos)
        excess = _excess_from_block_seed(self, i)
        count = 0

        if excess == min_v:
            count += 1
            if count == q:
                return i

        for pos in range(i + 1, j + 1):
            excess += -1 + (2 * self._b_ptr[pos])
            if excess == min_v:
                count += 1
                if count == q:
                    return pos

        return None

    cpdef SIZE_t nsibling(self, SIZE_t i) nogil:
        """The next sibling of i (i.e., the sibling to the right)

        nsibling(i) = close(i) + 1 (if the result j holds B[j] = 0 then i has no next sibling)

        Will return 0 if there is no sibling. This makes sense as the root
        cannot have a sibling by definition
        """
        cdef SIZE_t pos

        if self._b_ptr[i]:
            pos = self.close(i) + 1
        else:
            pos = self.nsibling(self.open(i))

        if pos >= self.size:
            return 0
        elif self._b_ptr[pos]:
            return pos
        else:
            return 0

    cpdef SIZE_t psibling(self, SIZE_t i) nogil:
        """The previous sibling of i (i.e., the sibling to the left)

        psibling(i) = open(i − 1) (if B[i − 1] = 1 then i has no previous sibling)

        Will return 0 if there is no sibling. This makes sense as the root
        cannot have a sibling by definition
        """
        cdef SIZE_t pos

        if self._b_ptr[i]:
            if self._b_ptr[max(0, i - 1)]:
                return 0

            pos = self.open(i - 1)
        else:
            pos = self.psibling(self.open(i))

        if pos < 0:
            return 0
        elif self._b_ptr[pos]:
            return pos
        else:
            return 0

    cpdef SIZE_t preorder(self, SIZE_t i) nogil:
        """Preorder rank of node i
        
        Parameters
        ----------
        i : int
            The node index to assess the preorder order of.

        Returns
        -------
        int
            The nodes order of evaluation in a preorder traversal of the tree.
        """
        if self._b_ptr[i]:
            return self.rank(1, i)
        else:
            return self.preorder(self.open(i))

    cpdef SIZE_t preorderselect(self, SIZE_t k) nogil:
        """The index of the node with preorder k
        
        Parameters
        ----------
        k : int
            The preorder evaluation order to search for.

        Returns
        -------
        int
            The index position of the node in the tree.
        """
        return self.select(1, k)

    cpdef SIZE_t postorder(self, SIZE_t i) nogil:
        """Postorder rank of node i
        
        Parameters
        ----------
        i : int
            The node index to assess the postorder order of.

        Returns
        -------
        int
            The nodes order of evaluation in a postorder traversal of the tree.
        """
        if self._b_ptr[i]:
            return self.rank(0, self.close(i))
        else:
            return self.rank(0, i)

    cpdef SIZE_t postorderselect(self, SIZE_t k) nogil:
        """The index of the node with postorder k
        
        Parameters
        ----------
        k : int
            The postorder evaluation order to search for.

        Returns
        -------
        int
            The index position of the node in the tree.
        """
        return self.open(self.select(0, k))

    cpdef BOOL_t isancestor(self, SIZE_t i, SIZE_t j) nogil:
        """Whether i is an ancestor of j

        Parameters
        ----------
        i : int
            A node index
        j : int
            A node index

        Note
        ----
        False is returned if i == j. A node cannot be an ancestor of itself.

        Returns
        -------
        bool
            True if i is an ancestor of j, False otherwise.
        """
        if i == j:
            return False

        if not self._b_ptr[i]:
            i = self.open(i)

        return i <= j < self.close(i)

    cpdef SIZE_t subtree(self, SIZE_t i) nogil:
        """The number of nodes in the subtree of i
        
        Parameters
        ----------
        i : int
            The node to evaluate

        Returns
        -------
        int
            The number of nodes in the subtree of i
        """
        if not self._b_ptr[i]:
            i = self.open(i)

        return (self.close(i) - i + 1) / 2

    cpdef SIZE_t levelancestor(self, SIZE_t i, SIZE_t d) nogil:
        """The ancestor j of i such that depth(j) = depth(i) − d
        
        Parameters
        ----------
        i : int
            The node to evaluate

        d : int
            How many ancestors back to evaluate

        Returns
        -------
        int
            The node index of the ancestor to search for
        """
        if d <= 0:
            return -1

        if not self._b_ptr[i]:
            i = self.open(i)

        return self.bwdsearch(i, -d - 1) + 1

    cpdef SIZE_t levelnext(self, SIZE_t i) nogil:
        """The next node with the same depth
        Parameters
        ----------
        i : int
            The node to evaluate

        Returns
        -------
        int
            The node index of the next node or -1 if there isn't one
        """
        return self.fwdsearch(self.close(i), 1)

    cpdef SIZE_t lca(self, SIZE_t i, SIZE_t j) nogil:
        """The lowest common ancestor of i and j

        Parameters
        ----------
        i : int
            A node index to evaluate
        j : int
            A node index to evalute

        Returns
        -------
        int
           The index of the lowest common ancestor
        """
        if self.isancestor(i, j):
            return i
        elif self.isancestor(j, i):
            return j
        else:
            return self.parent(self.rmq(i, j) + 1)

    cpdef SIZE_t deepestnode(self, SIZE_t i) nogil:
        """The index of the deepestnode which descends from i

        Parameters
        ----------
        i : int
            The node to evaluate

        Returns
        -------
        int
            The index of the deepest node which descends from i
        """
        return self.rMq(self.open(i), self.close(i))

    cpdef SIZE_t height(self, SIZE_t i) nogil:
        """The height of node i with respect to its deepest descendent

        Parameters
        ----------
        i : int
            The node to evaluate
        
        Notes
        -----
        Height is in terms of number of edges, not in terms of branch length
        
        Returns
        -------
        int
            The number of edges between node i and its deepest node
        """
        return self.excess(self.deepestnode(i)) - self.excess(self.open(i))

    cpdef BP shear(self, set tips):
        """Remove all nodes from the tree except tips and ancestors of tips

        Parameters
        ----------
        tips : set of str
            The set of tip names to retain

        Returns
        -------
        BP
            A new BP tree corresponding to only the described tips and their
            ancestors.
        """
        cdef:
            SIZE_t i, n = len(tips)
            SIZE_t p, t, count = 0
            BIT_ARRAY* mask
            BP new_bp

        mask = bit_array_create(self.B.size)
        bit_array_set_bit(mask, self.root())
        bit_array_set_bit(mask, self.close(self.root()))

        for i in range(self.B.size):
            # isleaf is only defined on the open parenthesis
            if self.isleaf(i):
                if self.name(i) in tips:  # gil is required for set operation
                    with nogil:
                        count += 1
                        bit_array_set_bit(mask, i)
                        bit_array_set_bit(mask, i + 1)

                        p = self.parent(i)
                        while p != 0 and bit_array_get_bit(mask, p) == 0:
                            bit_array_set_bit(mask, p)
                            bit_array_set_bit(mask, self.close(p))

                            p = self.parent(p)

        if count == 0:
            bit_array_free(mask)
            raise ValueError("No requested tips found")
                
        new_bp = self._mask_from_self(mask, self._lengths)
        bit_array_free(mask)
        return new_bp

    cdef BP _mask_from_self(self, BIT_ARRAY* mask, 
                            np.ndarray[DOUBLE_t, ndim=1] lengths):
        cdef:
            SIZE_t i, k, n, mask_sum
            np.ndarray[BOOL_t, ndim=1] new_b
            np.ndarray[object, ndim=1] new_names
            np.ndarray[object, ndim=1] names = self._names
            np.ndarray[DOUBLE_t, ndim=1] new_lengths
            BOOL_t* new_b_ptr
            DOUBLE_t* lengths_ptr
            DOUBLE_t* new_lengths_ptr

        n = bit_array_length(mask)
        mask_sum = bit_array_num_bits_set(mask)

        k = 0

        lengths_ptr = &lengths[0]

        new_b = np.empty(mask_sum, dtype=BOOL)
        new_names = np.empty(mask_sum, dtype=object)
        new_lengths = np.empty(mask_sum, dtype=DOUBLE)

        new_b_ptr = &new_b[0]
        new_lengths_ptr = &new_lengths[0]

        for i in range(n):
            if bit_array_get_bit(mask, i):
                new_b_ptr[k] = self._b_ptr[i]

                # since names is dtype=object, gil is required
                new_names[k] = names[i]
                new_lengths_ptr[k] = lengths_ptr[i]
                k += 1

        return BP(np.asarray(new_b), names=new_names, lengths=new_lengths)

    cpdef BP collapse(self):
        cdef:
            SIZE_t i, n = self.B.sum()
            SIZE_t current, first, last
            np.ndarray[DOUBLE_t, ndim=1] new_lengths
            BIT_ARRAY* mask
            DOUBLE_t* new_lengths_ptr
            BP new_bp
        
        mask = bit_array_create(self.B.size)
        bit_array_set_bit(mask, self.root())
        bit_array_set_bit(mask, self.close(self.root()))

        new_lengths = self._lengths.copy()
        new_lengths_ptr = <DOUBLE_t*>new_lengths.data

        with nogil:
            for i in range(n):
                current = self.preorderselect(i)

                if self.isleaf(current):
                    bit_array_set_bit(mask, current)
                    bit_array_set_bit(mask, self.close(current))
                else:
                    first = self.fchild(current)
                    last = self.lchild(current)

                    if first == last:
                        new_lengths_ptr[first] = new_lengths_ptr[first] + \
                                new_lengths_ptr[current]
                    else:
                        bit_array_set_bit(mask, current)
                        bit_array_set_bit(mask, self.close(current))

        new_bp = self._mask_from_self(mask, new_lengths)
        bit_array_free(mask)
        return new_bp

    cpdef inline SIZE_t ntips(self) nogil:
        cdef:
            SIZE_t i = 0
            SIZE_t count = 0
            SIZE_t n = self.size

        while i < (n - 1):
            if self._b_ptr[i] and not self._b_ptr[i+1]:
                count += 1
                i += 1
            i += 1

        return count

    cdef int scan_block_forward(self, int i, int k, int b, int d) nogil:
        """Scan a block forward from i

        Parameters
        ----------
        bp : BP
            The tree
        i : int
            The index position to start from in the tree
        k : int
            The block to explore
        b : int
            The block size
        d : int
            The depth to search for

        Returns
        -------
        int
            The index position of the result. -1 is returned if a result is not
            found.
        """
        cdef int lower_bound
        cdef int upper_bound
        cdef int j
        cdef int excess

        # lower_bound is block boundary or right of i
        lower_bound = max(k, 0) * b
        lower_bound = max(i + 1, lower_bound)

        # upper_bound is block boundary or end of tree
        upper_bound = min((k + 1) * b, self.size)

        if lower_bound > 0:
            excess = _excess_from_block_seed(self, lower_bound - 1)
        else:
            excess = 0

        for j in range(lower_bound, upper_bound):
            excess += -1 + (2 * self._b_ptr[j])
            if excess == d:
                return j
        
        return -1

    cdef int scan_block_backward(self, int i, int k, int b, int d) nogil:
        """Scan a block backward from i

        Parameters
        ----------
        i : int
            The index position to start from in the tree
        k : int
            The block to explore
        b : int
            The block size
        d : int
            The depth to search for

        Returns
        -------
        int
            The index position of the result. -1 is returned if a result is not
            found.
        """
        cdef int lower_bound
        cdef int upper_bound
        cdef int j
        cdef int excess
        
        # i and k are currently needed to handle the situation where 
        # k_start < i < k_end. It should be possible to resolve using partial 
        # excess.

        # range stop is exclusive, so need to set "stop" at -1 of boundary
        lower_bound = max(k, 0) * b - 1  # is it possible for k to be < 0?
        
        # include the right most position of the k-1 block so we can identify
        # closures spanning blocks. Not positive if this is correct, however if the
        # block is "()((", and we're searching for the opening paired with ")", 
        # we need to go to evaluate the excess prior to the first "(", at least as
        # "open" is defined in Cordova and Navarro
        if lower_bound >= 0:
            lower_bound -= 1
        
        # upper bound is block boundary or left of i, whichever is less
        # note that this is an inclusive boundary since this is a backward search
        upper_bound = min((k + 1) * b, self.size) - 1
        upper_bound = min(i - 1, upper_bound)
        
        if upper_bound <= 0:
            return -1

        excess = _excess_from_block_seed(self, upper_bound)
        for j in range(upper_bound, lower_bound, -1):
            if excess == d:
                return j
            excess -= -1 + (2 * self._b_ptr[j])

        return -1

    cdef SIZE_t fwdsearch(self, SIZE_t i, int d) nogil:
        """Search forward from i for desired excess

        Parameters
        ----------
        i : int
            The index to search forward from
        d : int
            The excess difference to search for (relative to E[i])
        
        Returns
        -------
        int
            The index of the result, or -1 if no result was found
        """
        cdef int k  # the block being interrogated
        cdef int result = -1 # the result of a scan within a block
        cdef int node  # the node within the binary tree being examined

        # get the block of parentheses to check
        k = i // self._rmm.b

        # desired excess
        d += _excess_from_block_seed(self, i)

        # determine which node our block corresponds too
        node = bt_node_from_left(k, self._rmm.height)

        # see if our result is in our current block
        if self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]:
            result = _scan_block_forward_rmm(self, i, k, d)

        # if we do not have a result, we need to begin traversal of the tree
        if result == -1:
            # walk up the tree
            while not bt_is_root(node):
                if bt_is_left_child(node):
                    node = bt_right_sibling(node)
                    if self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]:
                        break
                node = bt_parent(node)

            if bt_is_root(node):
                return -1

            # descend until we hit a leaf node
            while not bt_is_leaf(node, self._rmm.height):
                node = bt_left_child(node)

                # evaluate right, if not found, pick left
                if not (self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]):
                    node = bt_right_sibling(node)

            # we have found a block with contains our solution. convert from the
            # node index back into the block index
            k = node - <int>(pow(2, self._rmm.height) - 1)

            # scan for a result using the original d
            result = _scan_block_forward_rmm(self, i, k, d)

        return result

    cdef SIZE_t bwdsearch(self, SIZE_t i, int d) nogil:
        """Search backward from i for desired excess

        Parameters
        ----------
        i : int
            The index to search forward from
        d : int
            The excess difference to search for (relative to E[i])
        
        Returns
        -------
        int
            The index of the result, or -1 if no result was found
        """
        cdef int k  # the block being interrogated
        cdef int result = -1 # the result of a scan within a block
        cdef int node  # the node within the binary tree being examined
      
        # get the block of parentheses to check
        k = i // self._rmm.b  
        
        # desired excess
        d += _excess_from_block_seed(self, i)

        # see if our result is in our current block
        result = _scan_block_backward_rmm(self, i, k, d)

        # determine which node our block corresponds too
        node = bt_node_from_left(k, self._rmm.height)

        # special case: check sibling
        if result == -1 and bt_is_right_child(node):
            node = bt_left_sibling(node)
            k = node - <int>(pow(2, self._rmm.height) - 1)
            result = _scan_block_backward_rmm(self, i, k, d)
        
           # reset node and k in the event that result == -1
            k = i // self._rmm.b
            node = bt_right_sibling(node)
        
        # if we do not have a result, we need to begin traversal of the tree
        if result == -1:
            while not bt_is_root(node):
                # right nodes cannot contain the solution as we are searching left
                # As such, if we are the right node already, evaluate its sibling.
                if bt_is_right_child(node):
                    node = bt_left_sibling(node)
                    if self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]:
                        break
                
                # if we did not find a valid node, adjust for the relative
                # excess of the current node, and ascend to the parent
                node = bt_parent(node)
            
            if bt_is_root(node):
                return -1

            # descend until we hit a leaf node
            while not bt_is_leaf(node, self._rmm.height):
                node = bt_right_child(node)

                # evaluate right, if not found, pick left
                if not (self._rmm.mM[node, self._rmm.m_idx] <= d <= self._rmm.mM[node, self._rmm.M_idx]):
                    node = bt_left_sibling(node)

            # we have found a block with contains our solution. convert from the
            # node index back into the block index
            k = node - <int>(pow(2, self._rmm.height) - 1)

            # scan for a result
            result = _scan_block_backward_rmm(self, i, k, d)
            
        return result

# add in .r and .n into rmm calculation
#   - necessary for mincount/minselect
###
###
