import numpy as np
from typing import Union, Dict, Any, List

from pythermalcomfort.classes_input import IREQInputs
from pythermalcomfort.classes_return import IREQResult

def calc_ireq(M: Union[float, list, np.ndarray],
              W: Union[float, list, np.ndarray],
              ta: Union[float, list, np.ndarray],
              tr: Union[float, list, np.ndarray],
              p: Union[float, list, np.ndarray],
              w: Union[float, list, np.ndarray],
              v: Union[float, list, np.ndarray],
              rh: Union[float, list, np.ndarray],
              clo: Union[float, list, np.ndarray]) -> 'IREQResult':
    """
    Calculates the Required Clothing Insulation (IREQ) and Duration Limited Exposure (DLE) 
    based on ISO 11079.

    Parameters
    ----------
    M : float | list | np.ndarray
        Metabolic energy production, [W/m2]
    W : float | list | np.ndarray
        Rate of mechanical work, [W/m2]
    ta : float | list | np.ndarray
        Ambient air temperature, [<10 C]
    tr : float | list | np.ndarray
        Mean radiant temperature, [C]
    p : float | list | np.ndarray
        Air permeability, [l/m2s]
    w : float | list | np.ndarray
        Walking speed, [m/s]
    v : float | list | np.ndarray
        Relative air velocity, [0.4 to 18 m/s]
    rh : float | list | np.ndarray
        Relative humidity, [%]
    clo : float | list | np.ndarray
        Available basic clothing insulation, [clo]

    Returns
    -------
    IREQResult
        A dataclass containing the calculated IREQ, ICL, and DLE for both 
        minimal and neutral conditions.

    Applicability
    -------------
    * Metabolic rate: 58 to 400 W/m²
    * Ambient air temperature: <= 10 °C
    * Relative air velocity: 0.4 to 18 m/s
    * Relative humidity: 0 to 100%

    Raises
    ------
    ValueError
        If inputs are outside the valid domains (e.g., p <= 0, rh < 0).

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import calc_ireq
        
        result = calc_ireq(M=175.0, W=0.0, ta=-15.0, tr=-15.0, p=50.0, w=1.1, v=2.0, rh=55.0, clo=2.8)
        print(result.IREQminimal)

    References
    ----------
    ISO 11079:2007 Standard
    """
    # Check if inputs are all scalar to correctly format the output later
    is_scalar = np.isscalar(M) and np.isscalar(W) and np.isscalar(ta) and \
                np.isscalar(tr) and np.isscalar(p) and np.isscalar(w) and \
                np.isscalar(v) and np.isscalar(rh) and np.isscalar(clo)

    # Validate and normalize inputs through the input dataclass
    inp = IREQInputs(M=M, W=W, ta=ta, tr=tr, p=p, w=w, v=v, rh=rh, clo=clo)
    M_val, W_val, ta_val, tr_val, p_val, w_val, v_val, rh_val, clo_val = (
        inp.M, inp.W, inp.ta, inp.tr, inp.p, inp.w, inp.v, inp.rh, inp.clo
    )

    clo_m2cw = clo_val * 0.155
    Ia = 0.092 * np.exp(-0.15 * v_val - 0.22 * w_val) - 0.0045

    Tex = 29.0 + 0.2 * ta_val
    Pex = 0.1333 * np.exp(18.6686 - 4030.183 / (Tex + 235.0))
    Pa = (rh_val / 100.0) * 0.1333 * np.exp(18.6686 - 4030.183 / (ta_val + 235.0))
    
    ArAdu = 0.77
    results_dict = {}

    for calculation in [1, 2]:
        if calculation == 1:
            Tsk = 33.34 - 0.0354 * M_val
            wetness = np.full_like(M_val, 0.06)
        else:
            Tsk = 35.7 - 0.0285 * M_val
            wetness = 0.001 * M_val

        Psks = 0.1333 * np.exp(18.6686 - 4030.183 / (Tsk + 235.0))

        # --- 3. Vectorized Iteration to find IREQ ---
        IREQ = np.full_like(M_val, 0.5)
        factor = np.full_like(M_val, 0.5)
        balance = np.full_like(M_val, 1.0)

        for _ in range(150):
            active = np.abs(balance) > 0.01
            if not np.any(active):
                break

            fcl = 1.0 + 1.197 * IREQ
            Rt = (0.06 / 0.38) * (Ia + IREQ)
            E = wetness * (Psks - Pa) / Rt
            Hres = 1.73e-02 * M_val * (Pex - Pa) + 1.4e-03 * M_val * (Tex - ta_val)
            
            Tcl = Tsk - IREQ * (M_val - W_val - E - Hres)
            
            Tcl_K = 273.0 + Tcl
            tr_K = 273.0 + tr_val
            
            dT = Tcl - tr_val
            dT_safe = np.where(np.abs(dT) < 1e-4, 1e-4, dT)
            
            # Use np.where to handle zero division fallback
            hr_norm = (5.67e-08 * 0.95 * ArAdu * (Tcl_K**4 - tr_K**4)) / dT_safe
            hr_fallback = 5.67e-08 * 0.95 * ArAdu * 4 * (273.0 + (Tcl + tr_val) / 2.0)**3
            hr = np.where(np.abs(dT) < 1e-4, hr_fallback, hr_norm)
            
            hc = 1.0 / Ia - hr
            R = fcl * hr * (Tcl - tr_val)
            C = fcl * hc * (Tcl - ta_val)
            
            balance = M_val - W_val - E - Hres - R - C
            
            cond = balance > 0
            IREQ_new = np.where(cond, IREQ - factor, IREQ + factor)
            factor_new = np.where(cond, factor / 2.0, factor)
            
            IREQ = np.where(active, IREQ_new, IREQ)
            factor = np.where(active, factor_new, factor)

        IREQ_final = (Tsk - Tcl) / (R + C)

        # --- 4. Vectorized Iteration to find DLE ---
        Tcl_S = np.copy(ta_val)
        S = np.full_like(M_val, -40.0)
        factor_S = np.full_like(M_val, 500.0)
        Iclr = np.copy(clo_m2cw)
        balance_S = np.full_like(M_val, 1.0)
        
        for _ in range(150):
            active_S = np.abs(balance_S) > 0.01
            if not np.any(active_S):
                break
                
            fcl_S = 1.0 + 1.197 * Iclr
            Iclr = ((clo_m2cw + 0.085 / fcl_S) * (0.54 * np.exp(-0.15 * v_val - 0.22 * w_val) * (p_val**0.075) - 0.06 * np.log(p_val) + 0.5) -
                    (0.092 * np.exp(-0.15 * v_val - 0.22 * w_val) - 0.0045) / fcl_S)
            
            Rt = (0.06 / 0.38) * (Ia + Iclr)
            E = wetness * (Psks - Pa) / Rt
            Hres = 1.73e-02 * M_val * (Pex - Pa) + 1.4e-03 * M_val * (Tex - ta_val)
            
            Tcl_S = Tsk - Iclr * (M_val - W_val - E - Hres - S)
            
            Tcl_K = 273.0 + Tcl_S
            tr_K = 273.0 + tr_val
            
            dT = Tcl_S - tr_val
            dT_safe = np.where(np.abs(dT) < 1e-4, 1e-4, dT)
            hr_norm_S = (5.67e-08 * 0.95 * ArAdu * (Tcl_K**4 - tr_K**4)) / dT_safe
            hr_fallback_S = 5.67e-08 * 0.95 * ArAdu * 4 * (273.0 + (Tcl_S + tr_val) / 2.0)**3
            hr_S = np.where(np.abs(dT) < 1e-4, hr_fallback_S, hr_norm_S)
            
            hc_S = 1.0 / Ia - hr_S
            R_S = fcl_S * hr_S * (Tcl_S - tr_val)
            C_S = fcl_S * hc_S * (Tcl_S - ta_val)
            
            balance_S = M_val - W_val - E - Hres - R_S - C_S - S
            
            cond_S = balance_S > 0
            S_new = np.where(cond_S, S + factor_S, S - factor_S)
            factor_S_new = np.where(cond_S, factor_S / 2.0, factor_S)
            
            S = np.where(active_S, S_new, S)
            factor_S = np.where(active_S, factor_S_new, factor_S)

        with np.errstate(divide='ignore'):
            DLE = -40.0 / S
        
        # --- 5. Store and format results ---
        constant_part = 0.54 * np.exp(-0.15 * v_val - 0.22 * w_val) * (p_val**0.075) - 0.06 * np.log(p_val) + 0.5
        
        IREQ_out = np.round((IREQ_final / 0.155) * 10.0) / 10.0
        
        fcl_final = 1.0 + 1.197 * IREQ_final
        ICL_raw = ((IREQ_final + Ia / fcl_final) / constant_part - 0.085 / fcl_final)
        ICL_out = np.round((ICL_raw / 0.155) * 10.0) / 10.0
        
        DLE_out = np.empty(DLE.shape, dtype=object)
        condition = (DLE > 8.0) | (DLE < 0)
        DLE_out[condition] = "more than 8"
        DLE_out[~condition] = np.round(DLE[~condition], 1)

        # Unpack back to scalar if input was scalar
        if is_scalar:
            IREQ_out = float(IREQ_out[0])
            ICL_out = float(ICL_out[0])
            DLE_out = DLE_out[0]
        else:
            DLE_out = DLE_out.tolist()

        postfix = "minimal" if calculation == 1 else "neutral"
        results_dict[f"IREQ{postfix}"] = IREQ_out
        results_dict[f"ICL{postfix}"] = ICL_out
        results_dict[f"DLE{postfix}"] = DLE_out

    return IREQResult(
        IREQminimal=results_dict["IREQminimal"],
        ICLminimal=results_dict["ICLminimal"],
        DLEminimal=results_dict["DLEminimal"],
        IREQneutral=results_dict["IREQneutral"],
        ICLneutral=results_dict["ICLneutral"],
        DLEneutral=results_dict["DLEneutral"]
    )
