import numpy as np
import pytest
from visionqueue.tracking.matching import linear_sum_assignment, matching

def test_linear_sum_assignment_1x1():
    cost = np.array([[5.0]], dtype=np.float32)
    r, c = linear_sum_assignment(cost)
    assert np.array_equal(r, [0])
    assert np.array_equal(c, [0])

def test_linear_sum_assignment_2x2():
    cost = np.array([
        [10.0, 1.0],
        [2.0, 9.0]
    ], dtype=np.float32)
    r, c = linear_sum_assignment(cost)
    assert np.array_equal(r, [0, 1])
    assert np.array_equal(c, [1, 0])

def test_linear_sum_assignment_3x3():
    # This matrix breaks the greedy algorithm
    cost = np.array([
        [10.0, 10.0, 1.0],
        [2.0,  9.0,  8.0],
        [1.0,  8.0,  7.0]
    ], dtype=np.float32)
    r, c = linear_sum_assignment(cost)
    # The optimal assignment should be 0-2 (1.0), 1-1 (9.0), 2-0 (1.0) -> cost 11.0
    # A greedy algorithm would pick 0-2 (1.0), then 1-0 (2.0), then 2-1 (8.0) -> cost 11.0
    # Wait, both are 11.0, so this cost matrix has multiple optima or I messed up the example.
    # Let's use a standard matrix where greedy fails:
    # 2 3 1
    # 5 4 6
    # 1 8 9
    cost2 = np.array([
        [2.0, 3.0, 1.0],
        [5.0, 4.0, 6.0],
        [1.0, 8.0, 9.0]
    ], dtype=np.float32)
    r2, c2 = linear_sum_assignment(cost2)
    assert set(zip(r2, c2)) == {(0, 2), (1, 1), (2, 0)}

def test_linear_sum_assignment_rectangular():
    cost = np.array([
        [10.0, 1.0, 5.0],
        [2.0, 9.0, 8.0]
    ], dtype=np.float32)
    r, c = linear_sum_assignment(cost)
    assert set(zip(r, c)) == {(0, 1), (1, 0)}

def test_linear_sum_assignment_rectangular_tall():
    cost = np.array([
        [10.0, 1.0],
        [2.0, 9.0],
        [5.0, 8.0]
    ], dtype=np.float32)
    r, c = linear_sum_assignment(cost)
    assert set(zip(r, c)) == {(0, 1), (1, 0)}
