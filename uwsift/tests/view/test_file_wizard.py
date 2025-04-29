import pytest

from uwsift.view.open_file_wizard import NumericTableWidgetItem


@pytest.mark.parametrize(
    "lower, higher, exp_result",
    [
        (1, 2, True),
        (10000000, 10000001, True),
        (2, 1, False),
        (3, 2, False),
        (4, 4, False),
        (0.001, 0.01, True),
        (0.1, 0.01, False),
        (0.00001, None, True),
    ],
)
def test_numeric_table_widget_item(lower, higher, exp_result):
    ntwi_lower = NumericTableWidgetItem(f"{lower} µm", lower)
    ntwi_higher = NumericTableWidgetItem(f"{higher} µm", higher)
    assert (ntwi_lower < ntwi_higher) is exp_result
