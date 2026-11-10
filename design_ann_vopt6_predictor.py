#!/usr/bin/env python3
"""Corrected nominal-design ANN with algebraic source/energy constraints.

The projection enforces declared balances; it is not a thermodynamic solver.
Runtime dependencies are limited to the packaged NumPy/pandas/sklearn stack.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

SMALL = 1.0e-12
DUAL_STAGE_DESIGN_PRESSURE_RATIO_THRESHOLD = 8.0


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    return str(obj)


def brine_enthalpy(table: dict, pressure_bar, temperature_C) -> np.ndarray:
    """Reference bilinear PT interpolation and its linear extrapolation."""
    p, t = np.broadcast_arrays(np.asarray(pressure_bar, float), np.asarray(temperature_C, float)+273.15)
    pg, tg, values = (np.asarray(table[k], float) for k in ("p_bar", "T_K", "h_J_kg"))
    i = np.clip(np.searchsorted(pg, p)-1, 0, len(pg)-2)
    j = np.clip(np.searchsorted(tg, t)-1, 0, len(tg)-2)
    wp, wt = (p-pg[i])/(pg[i+1]-pg[i]), (t-tg[j])/(tg[j+1]-tg[j])
    return ((1-wp)*(1-wt)*values[i,j]+wp*(1-wt)*values[i+1,j]
            +(1-wp)*wt*values[i,j+1]+wp*wt*values[i+1,j+1])


def saturation_pressure(approx: dict, temperature_C) -> np.ndarray:
    """Bounded log-pressure Chebyshev approximation fitted to CoolProp."""
    t = np.asarray(temperature_C, float)
    lo, hi = map(float, approx["domain_C"])
    if np.any(~np.isfinite(t)) or np.any(t < lo-1e-9) or np.any(t > hi+1e-9):
        raise ValueError(f"Saturation pressure requires {lo:g} <= T_C <= {hi:g}")
    return np.exp(np.polynomial.chebyshev.chebval((2*t-hi-lo)/(hi-lo), approx["log_bar_coefficients"]))


class DesignANNModule:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self.bundle = joblib.load(self.model_path)
        if self.bundle.get("schema_version") != "corrected_design_ann_v1":
            raise ValueError("This predictor requires the corrected design ANN bundle")
        self.base_inputs = list(self.bundle["base_input_columns"])
        self.feature_columns = list(self.bundle["feature_columns"])
        self.learned_outputs = list(self.bundle["learned_output_columns"])
        self.report_outputs = list(self.bundle["report_output_columns"])
        self.nominal_flow = float(self.bundle["nominal_module_brine_flow_kg_s"])
        self.hp_drop = float(self.bundle["hp_brine_inlet_drop_K"])
        self.hp_out_min = float(self.bundle["hp_brine_out_min_C"])
        self.constants = dict(self.bundle["constants"])

    def build_features(self, base: pd.DataFrame) -> pd.DataFrame:
        out = base[self.base_inputs].astype(float).copy()
        out["source_deltaT_available_K"] = out.T_brine_in_C-out.T_brine_out_min_C
        out["source_capacity_to_target_ratio"] = 4180.*self.nominal_flow*out.source_deltaT_available_K/out.target_load_W.clip(lower=SMALL)
        out["brine_in_minus_ambient_K"] = out.T_brine_in_C-out.avg_T_amb_C
        out["brine_out_min_minus_ambient_K"] = out.T_brine_out_min_C-out.avg_T_amb_C
        return out[self.feature_columns]

    def domain_warnings(self, row: pd.Series) -> list[str]:
        messages = []
        for col, lim in self.bundle["input_bounds"].items():
            value, lo, hi = float(row[col]), float(lim["min"]), float(lim["max"])
            if value < lo or value > hi:
                messages.append(f"{col}={value:.6g} outside declared design domain [{lo:.6g}, {hi:.6g}]")
        messages.append("Predicted geometry is nominal sizing; fixed 1 MW off-design bundles describe their own saved hardware.")
        return messages

    def _base(self, base: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.base_inputs if c not in base]
        if missing: raise KeyError(f"Missing ANN input columns: {missing}")
        result = base[self.base_inputs].astype(float).reset_index(drop=True)
        if not np.isfinite(result.to_numpy()).all(): raise ValueError("ANN inputs must be finite")
        if (result.target_load_W <= 0).any(): raise ValueError("target_load_W must be positive")
        if (result.T_brine_in_C-self.hp_drop <= self.hp_out_min).any():
            raise ValueError("HP inlet must exceed its minimum outlet temperature")
        return result

    def _routing(self, base: pd.DataFrame) -> dict[str, np.ndarray]:
        te = base.T_brine_in_C.to_numpy()-10.
        tc = base.avg_T_amb_C.to_numpy()+float(self.constants["orc_cond_ambient_offset_K"])
        approx = self.bundle["saturation_pressure_approximation"]
        pe, pc = saturation_pressure(approx, te), saturation_pressure(approx, tc)
        ratio = pe/pc
        return {"orc_T_evap_C": te, "orc_T_cond_C": tc, "orc_p_evap_bar": pe,
                "orc_p_cond_bar": pc, "orc_design_pressure_ratio": ratio,
                "orc_expander_stage_count": np.where(ratio > 8., 2, 1)}

    def predict_raw_frame(self, base: pd.DataFrame) -> pd.DataFrame:
        """Learned positive quantities before projection, provided for auditing."""
        base = self._base(base)
        route, features = self._routing(base), self.build_features(base)
        out = pd.DataFrame(0., index=base.index, columns=self.learned_outputs)
        for number, expert in self.bundle["stage_models"].items():
            mask = route["orc_expander_stage_count"] == int(number)
            if not mask.any(): continue
            x = expert["x_scaler"].transform(features.loc[mask].to_numpy())
            pred = expert["y_scaler"].inverse_transform(expert["regressor"].predict(x))
            load_scale = base.loc[mask, "target_load_W"].to_numpy()/1e6
            for j, col in enumerate(expert["output_columns"]):
                value = pred[:,j]
                if col in expert["log_columns"]: value = np.exp(np.clip(value, -40., 40.))
                if col in expert["load_scaled_columns"]: value = value*load_scale
                out.loc[mask,col] = value
        return out

    def _reconstruct(self, base: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
        out = learned.copy()
        target, pressure, tin = (base[c].to_numpy() for c in ("target_load_W", "p_brine_bar", "T_brine_in_C"))
        table = self.bundle["brine_enthalpy_table"]
        hp_in = tin-self.hp_drop
        hin, hmin = brine_enthalpy(table, pressure, hp_in), brine_enthalpy(table, pressure, self.hp_out_min)
        available = np.maximum(self.nominal_flow*(hin-hmin), 0.)
        q = np.clip(out.hp_Q_cond_W.to_numpy(), SMALL, target)
        w = np.clip(out.hp_W_comp_W.to_numpy(), SMALL, q)
        source = q-w
        scale = np.minimum(1., available/np.maximum(source, SMALL))
        q, w, source = q*scale, w*scale, source*scale
        out["hp_Q_cond_W"], out["hp_W_comp_W"] = q, w
        out["hp_source_projection_scale"] = scale
        out["hp_m_dot_ref_kg_s"] = np.maximum(out.hp_m_dot_ref_kg_s, SMALL)*scale
        out["hp_W_comp_nominal_required_W"] = np.clip(out.hp_W_comp_nominal_required_W, SMALL, target)
        out["hp_Q_cond_nominal_W"], out["hp_Q_cond_achievable_W"] = target, q
        out["hp_Q_evap_W"] = source
        out["hp_Q_evap_nominal_required_W"] = target-out.hp_W_comp_nominal_required_W
        out["hp_W_comp_operating_W"] = w
        out["hp_COP_heating"] = q/np.maximum(w, SMALL)
        out["hp_unmet_load_W"] = np.maximum(target-q, 0.)
        out["hp_load_coverage_fraction"] = q/target
        out["hp_T_brine_in_C"], out["hp_T_brine_out_min_C"] = hp_in, self.hp_out_min
        goal = hin-source/self.nominal_flow
        lo, hi = np.full(len(base), self.hp_out_min), hp_in.copy()
        for _ in range(42):
            mid = (lo+hi)/2
            low = brine_enthalpy(table, pressure, mid) < goal
            lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
        out["hp_T_brine_out_C"] = (lo+hi)/2
        out["hp_T_evap_C"] = out.hp_T_brine_out_C-5.
        approx = self.bundle["saturation_pressure_approximation"]
        out["hp_p_evap_bar"] = saturation_pressure(approx, out.hp_T_evap_C)
        out["hp_p_cond_bar"] = saturation_pressure(approx, out.hp_T_cond_C)
        out["hp_source_margin_at_nominal"] = available/np.maximum(out.hp_Q_evap_nominal_required_W, SMALL)
        out["source_limited_nominal"] = out.hp_source_margin_at_nominal < .999
        out["source_energy_residual_W"] = source-self.nominal_flow*(hin-brine_enthalpy(table, pressure, out.hp_T_brine_out_C))
        out["hp_energy_balance_residual_W"] = q-source-w
        for key, value in self._routing(base).items(): out[key] = value
        orc_available = np.maximum(self.nominal_flow*(brine_enthalpy(table, pressure, tin)
                                 -brine_enthalpy(table, pressure, base.T_brine_out_min_C)), 0.)
        orc_scale = np.minimum(1., orc_available/np.maximum(out.orc_Q_in_W.to_numpy(), SMALL))
        for col in ["orc_Q_in_W", "orc_P_gross_W", "orc_P_pump_el_W", "orc_P_acc_fan_W", "orc_P_other_aux_W",
                    "orc_m_dot_orc_kg_s", "orc_V_s_exp_stage1_m3rev", "orc_V_s_exp_stage2_m3rev"]:
            out[col] = np.maximum(out[col], 0.)*orc_scale
        out["orc_source_projection_scale"] = orc_scale
        out["orc_P_exp_el_W"] = float(self.constants["eta_generator"])*out.orc_P_gross_W
        out["orc_P_net_W"] = out.orc_P_exp_el_W-out.orc_P_pump_el_W-out.orc_P_acc_fan_W-out.orc_P_other_aux_W
        out["orc_Q_rej_W"] = out.orc_Q_in_W+float(self.constants["eta_pump_motor"])*out.orc_P_pump_el_W-out.orc_P_gross_W
        out["orc_eta_gross"] = out.orc_P_gross_W/out.orc_Q_in_W.clip(lower=SMALL)
        out["orc_eta_net"] = out.orc_P_net_W/out.orc_Q_in_W.clip(lower=SMALL)
        out["orc_V_s_exp_m3rev"] = out.orc_V_s_exp_stage1_m3rev
        out["orc_expander_fallback_probability"] = 0.
        out["orc_electrical_balance_residual_W"] = out.orc_P_net_W-(out.orc_P_exp_el_W-out.orc_P_pump_el_W-out.orc_P_acc_fan_W-out.orc_P_other_aux_W)
        out["common_hex_UA_required_W_K"] = float(self.constants["common_hex_U_W_m2K"])*out.common_hex_area_req_m2
        out["common_hex_orc_fits_in_hp_common_hx"] = out.common_hex_orc_area_req_m2 <= out.common_hex_area_req_m2*(1+1e-9)
        out["acc_UA_required_W_K"] = float(self.constants["acc_U_W_m2K"])*out.acc_area_req_m2
        out["design_Q_geo_design_W"] = available
        out["design_m_brine_kg_s"] = self.nominal_flow
        return out[self.report_outputs]

    def predict_frame(self, base: pd.DataFrame) -> pd.DataFrame:
        base = self._base(base)
        return self._reconstruct(base, self.predict_raw_frame(base))

    def predict_one(self, *, target_load_W: float, avg_T_amb_C: float, T_brine_in_C: float,
                    p_brine_bar: float, T_brine_out_min_C: float) -> dict[str, Any]:
        base = pd.DataFrame([dict(target_load_W=float(target_load_W), avg_T_amb_C=float(avg_T_amb_C),
            T_brine_in_C=float(T_brine_in_C), p_brine_bar=float(p_brine_bar), T_brine_out_min_C=float(T_brine_out_min_C))])
        pred = self.predict_frame(base).iloc[0]
        return {"inputs": base.iloc[0].to_dict(), "module_design": pred.to_dict(), "warnings": self.domain_warnings(base.iloc[0])}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict a corrected nominal 10 kg/s reversible-module design.")
    p.add_argument("--model", default=str(Path(__file__).parent/"models"/"design_ann_vopt6_module.joblib"))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-load-W", type=float)
    group.add_argument("--target-load-mw", type=float)
    p.add_argument("--avg-T-amb-c", type=float, required=True)
    p.add_argument("--T-brine-in-c", type=float, required=True)
    p.add_argument("--p-brine-bar", type=float, required=True)
    p.add_argument("--T-brine-out-min-c", type=float, required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    target = args.target_load_W if args.target_load_W is not None else args.target_load_mw*1e6
    result = DesignANNModule(args.model).predict_one(target_load_W=target, avg_T_amb_C=args.avg_T_amb_c,
        T_brine_in_C=args.T_brine_in_c, p_brine_bar=args.p_brine_bar, T_brine_out_min_C=args.T_brine_out_min_c)
    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__": main()
