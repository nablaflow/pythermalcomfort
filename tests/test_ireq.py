import numpy as np
import pytest

from pythermalcomfort.models.ireq import calc_ireq


def test_ireq_scalar():
    """Test scalar inputs matching standard website validation."""
    # These parameters match the screenshot validation you provided
    result = calc_ireq(
        M=175.0,
        W=0.0,
        ta=-15.0,
        tr=-15.0,
        p=50.0,
        w=1.1,
        v=2.0,
        rh=55.0,
        clo=2.8
    )

    # IREQ minimal values
    assert pytest.approx(result.IREQminimal, abs=0.1) == 1.6
    assert pytest.approx(result.ICLminimal, abs=0.1) == 2.2
    assert result.DLEminimal == "more than 8"

    # IREQ neutral values
    assert pytest.approx(result.IREQneutral, abs=0.1) == 1.9
    assert pytest.approx(result.ICLneutral, abs=0.1) == 2.6
    assert result.DLEneutral == "more than 8"


def test_ireq_array():
    """Test standard array calculations for IREQ."""
    params_array = {
        "M": [175.0, 116.0],
        "W": [0.0, 0.0],
        "ta": [-15.0, -10.0],
        "tr": [-15.0, -10.0],
        "p": [50.0, 8.0],
        "w": [1.1, 0.3],
        "v": [2.0, 0.5],
        "rh": [55.0, 50.0],
        "clo": [2.8, 1.5],
    }

    result = calc_ireq(**params_array)
    
    # Assert return types and shapes
    assert isinstance(result.IREQminimal, (list, np.ndarray))
    assert len(result.IREQminimal) == 2
    assert isinstance(result.DLEminimal, list)
    
    # Assert values for the first index are identical to scalar test
    assert pytest.approx(result.IREQminimal[0], abs=0.1) == 1.6
    assert result.DLEminimal[0] == "more than 8"


def test_ireq_boundaries():
    """Test auto-bounding limits for IREQ inputs according to ISO 11079."""
    # Sending out-of-bounds parameters like M=10 or ta=50 to test if __post_init__ clamps them
    result_out_of_bounds = calc_ireq(
        M=10.0,     # Should clamp to 58.0
        W=0.0,
        ta=50.0,    # Should clamp to 10.0
        tr=10.0,
        p=50.0,
        w=0.0,      # Should clamp to calculated min
        v=0.1,      # Should clamp to 0.4
        rh=50.0,
        clo=1.5
    )
    
    # Sending the explicitly clamped equivalent parameters to compare
    result_clamped = calc_ireq(
        M=58.0, 
        W=0.0,
        ta=10.0,
        tr=10.0,
        p=50.0,
        w=0.0,      # min logic is based on (58-58) = 0.0
        v=0.4,
        rh=50.0,
        clo=1.5
    )
    
    assert pytest.approx(result_out_of_bounds.IREQminimal, abs=0.01) == result_clamped.IREQminimal