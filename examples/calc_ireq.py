import sys
import os

# Adjust path to allow local imports if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pythermalcomfort.models.ireq import calc_ireq

if __name__ == "__main__":
    # Test 1: Single Scalar Input (matching your previous test)
    params = {
        "M": 175.0,
        "W": 0.0,
        "ta": -15.0,
        "tr": -15.0,
        "p": 50.0,
        "w": 1.1,
        "v": 2.0,
        "rh": 55.0,
        "clo": 2.8
    }
    
    print("--- [Test 1] Scalar Input Results ---")
    res_scalar = calc_ireq(**params)
    print(res_scalar)
    
    # Test 2: Multi-value Vectorized Input (e.g. an entire Excel column)
    params_array = {
        "M": [175.0, 116.0],
        "W": [0.0, 0.0],
        "ta": [-15.0, -10.0],
        "tr": [-15.0, -10.0],
        "p": [50.0, 8.0],
        "w": [1.1, 0.3],
        "v": [2.0, 0.5],
        "rh": [55.0, 50.0],
        "clo": [2.8, 1.5]
    }
    print("\n--- [Test 2] Vectorized Array Input Results (List processing) ---")
    res_array = calc_ireq(**params_array)
    print(res_array)
