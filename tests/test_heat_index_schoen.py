import numpy as np
import pytest

from pythermalcomfort.classes_return import HI
from pythermalcomfort.models.heat_index_schoen import heat_index_schoen
from pythermalcomfort.utilities import psy_ta_rh
from tests.conftest import is_equal


def _expected_heat_index_schoen(tdb, rh):
    """Calculate Schoen heat index from the published equation."""
    tdb = np.asarray(tdb)
    rh = np.asarray(rh)
    t_dew = psy_ta_rh(tdb, rh, p_atm=101325).dew_point_tmp
    return tdb - 1.0799 * np.exp(0.03755 * tdb) * (1 - np.exp(0.0801 * (t_dew - 14)))


def _expected_stress_category(hi):
    """Calculate heat stress categories from heat index thresholds."""
    hi = np.asarray(hi)
    categories = np.full_like(hi, "extreme danger", dtype=object)
    categories[hi <= 54] = "danger"
    categories[hi <= 41] = "extreme caution"
    categories[hi <= 32] = "caution"
    categories[hi <= -1000] = "no risk"
    categories[hi > 1000] = np.nan
    return categories


def test_scalar_rounding_default() -> None:
    """Test that Schoen heat index returns a rounded scalar result by default."""
    result = heat_index_schoen(tdb=29, rh=50)
    expected = np.around(_expected_heat_index_schoen(29, 50), 1)

    assert isinstance(result, HI)
    assert result.hi.shape == ()
    assert is_equal(result.hi, expected)
    assert result.stress_category == "caution"


def test_scalar_no_rounding() -> None:
    """Test that round_output=False preserves the unrounded result."""
    result = heat_index_schoen(tdb=29, rh=50, round_output=False)
    expected = _expected_heat_index_schoen(29, 50)

    assert is_equal(result.hi, expected)


def test_list_input() -> None:
    """Test that Schoen heat index supports vector inputs."""
    tdb = [29, 35, 45, 55]
    rh = [50, 60, 70, 80]

    result = heat_index_schoen(tdb=tdb, rh=rh)
    expected = np.around(_expected_heat_index_schoen(tdb, rh), 1)

    assert isinstance(result.hi, np.ndarray)
    assert result.hi.shape == (4,)
    assert is_equal(result.hi, expected)
    assert is_equal(result.stress_category, _expected_stress_category(expected))


@pytest.mark.parametrize(
    ("tdb", "rh", "expected_error"),
    [
        ("29", 50, TypeError),
        (29, "50", TypeError),
        (29, -1, ValueError),
        (29, 101, ValueError),
    ],
)
def test_invalid_inputs_raise_specific(tdb, rh, expected_error) -> None:
    """Test that invalid input types and humidity values raise errors."""
    with pytest.raises(expected_error):
        heat_index_schoen(tdb=tdb, rh=rh)
