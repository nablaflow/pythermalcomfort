import pprint
from pythermalcomfort.models import calc_ireq

def main():
    print("-------------------------------------------------------------")
    print("Example 1: Single values calculation (ISO 11079)")
    print("-------------------------------------------------------------")
    result_scalar = calc_ireq(
        M=175.0,     # Metabolic energy production (W/m2)
        W=0.0,       # Rate of mechanical work (W/m2)
        ta=-15.0,    # Ambient air temperature (C)
        tr=-15.0,    # Mean radiant temperature (C)
        p=50.0,      # Air permeability (l/m2s)
        w=1.1,       # Walking speed (m/s)
        v=2.0,       # Relative air velocity (m/s)
        rh=55.0,     # Relative humidity (%)
        clo=2.8      # Available basic clothing insulation (clo)
    )
    pprint.pprint(result_scalar)
    
    print("\n-------------------------------------------------------------")
    print("Example 2: Equivalent calculation with vectorized array inputs")
    print("-------------------------------------------------------------")
    result_array = calc_ireq(
        M=[175.0, 116.0],
        W=[0.0, 0.0],
        ta=[-15.0, -10.0],
        tr=[-15.0, -10.0],
        p=[50.0, 8.0],
        w=[1.1, 0.3],
        v=[2.0, 0.5],
        rh=[55.0, 50.0],
        clo=[2.8, 1.5],
    )
    pprint.pprint(result_array)

if __name__ == '__main__':
    main()
