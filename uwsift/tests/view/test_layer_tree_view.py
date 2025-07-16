#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test layer tree view."""
import pytest

from uwsift.view.layer_tree_view import EqualizerBarDelegate

NUM_TEST_RANGES = 5


@pytest.fixture(scope="session")
def empty_ebd():
    empty_ebd = EqualizerBarDelegate()
    empty_ebd._reset()
    return empty_ebd


@pytest.fixture(scope="function")
def ebd():
    ebd = EqualizerBarDelegate()
    ebd._reset()
    for index in range(NUM_TEST_RANGES):
        ebd._process_range(index, (index, index))
    return ebd


@pytest.mark.parametrize(
    "index, min_max_tuple, min_max_tuple_exp",
    [
        (0, (-1.0, 2.0), (-1.0, 2.0)),
        (1, (-1.1, 2.2), (-1.1, 2.2)),
        (1, (-1.0, 2.1), (-1.1, 2.2)),
        (1, (-1.2, 2.2), (-1.2, 2.2)),
        (1, (-1.1, 2.3), (-1.2, 2.3)),
        (1, (2.4, -1.3), (-1.3, 2.4)),
    ],
)
def test_equalizer_bar_delegate_total_ranges(empty_ebd, index, min_max_tuple, min_max_tuple_exp):
    empty_ebd._process_range(index, min_max_tuple)
    assert empty_ebd._total_ranges[index][0] == min_max_tuple_exp[0]
    assert empty_ebd._total_ranges[index][1] == min_max_tuple_exp[1]


@pytest.mark.parametrize(
    "index_to_move, index_to_insert_before, exp_tuple_seq",
    [
        (3, 1, [(0, 0), (3, 3), (1, 1), (2, 2), (4, 4)]),
        (4, 0, [(4, 4), (0, 0), (1, 1), (2, 2), (3, 3)]),
        (1, 3, [(0, 0), (2, 2), (1, 1), (3, 3), (4, 4)]),
        (2, 5, [(0, 0), (1, 1), (3, 3), (4, 4), (2, 2)]),
    ],
)
def test_layer_reordering(ebd, index_to_move, index_to_insert_before, exp_tuple_seq):
    ebd._layers_reordered(None, index_to_move, index_to_move, None, index_to_insert_before)
    sorted_layer_indices = sorted(ebd._total_ranges.keys())
    for index in sorted_layer_indices:
        first = exp_tuple_seq[index]
        second = ebd._total_ranges[index]
        assert first == second
