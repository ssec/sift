from decimal import Decimal

import numpy as np
import pytest

from uwsift.util.common import format_resolution


@pytest.mark.parametrize(
    "resolution, expected",
    [
        (999, "999 m"),  # Test with a value less than 1000 meters
        (1000, "1.0 km"),  # Test with exactly 1000 meters
        (1500, "1.5 km"),  # Test with a value greater than 1000 meters
        (25000, "25.0 km"),  # Test with a large value
        (1, "1 m"),  # Test with a value close to 1 meter
        (1234.56, "1.2 km"),  # Test with a float value
        (np.float32(500), "500 m"),  # Test with a numpy numeric value less than 1000 meters
        (np.float64(1500), "1.5 km"),  # Test with a numpy numeric value greater than 1000 meters
        (Decimal("750.0"), "750 m"),  # Test with a Decimal value less than 1000 meters
        (Decimal("1250.0"), "1.2 km"),  # Test with a Decimal value greater than 1000 meters
    ],
)
def test_format_resolution(resolution, expected):
    assert format_resolution(resolution) == expected
