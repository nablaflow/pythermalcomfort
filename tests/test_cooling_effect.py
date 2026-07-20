import numpy as np

from pythermalcomfort.models import cooling_effect
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_cooling_effect_regression_values() -> None:
    """Pin ce against the reference (pre-numba) implementation.

    Values were captured from the implementation before its inner SET
    calculation was rewritten to call the numba-jitted Gagge kernel directly
    (instead of going through the full set_tmp()/two_nodes_gagge() public API
    on every root-finding iteration). Confirmed to match bit-for-bit across a
    wide random sweep plus edge cases (vr <= 0.1, vr == 0.1 boundary).
    """
    cases = [
        (25, 25, 0.05, 50, 1.2, 0.5, 0.0),  # vr <= 0.1 -> 0
        (25, 25, 0.1, 50, 1.2, 0.5, 0.0),  # boundary
        (25, 25, 0.3, 50, 1.2, 0.5, 1.68),
        (35, 35, 1.5, 90, 2.5, 0.2, 5.14),
        (15, 15, 0.5, 10, 1.0, 1.2, 2.61),
    ]
    for tdb, tr, vr, rh, met, clo, expected in cases:
        ce = cooling_effect(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo).ce
        assert np.isclose(ce, expected, atol=0.01), (
            tdb,
            tr,
            vr,
            rh,
            met,
            clo,
            ce,
            expected,
        )


def test_cooling_effect(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the cooling effect correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.COOLING_EFFECT.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = cooling_effect(**inputs)

        validate_result(result, outputs, tolerance)
