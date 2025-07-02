#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test layer tree view."""
import pytest

from uwsift.view.layer_tree_view import EqualizerBarDelegate


#
@pytest.fixture(scope="module")
def ebd():
    return EqualizerBarDelegate()


@pytest.mark.parametrize(
    "index,min_max_tuple, min_max_tuple_exp",
    [
        (0, (-1.0, 2.0), (-1.0, 2.0)),
        (1, (-1.1, 2.2), (-1.1, 2.2)),
        (1, (-1.0, 2.1), (-1.1, 2.2)),
        (1, (-1.2, 2.2), (-1.2, 2.2)),
        (1, (-1.1, 2.3), (-1.2, 2.3)),
        (1, (2.4, -1.3), (-1.3, 2.4)),
    ],
)
def test_equalizer_bar_delegate_climits_map(ebd, index, min_max_tuple, min_max_tuple_exp):
    ebd._eval_and_store_climit(index, min_max_tuple)
    assert ebd._climits_map[index][0] == min_max_tuple_exp[0]
    assert ebd._climits_map[index][1] == min_max_tuple_exp[1]
