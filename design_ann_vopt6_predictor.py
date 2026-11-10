#!/usr/bin/env python3
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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class DesignANNModule:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).resolve()
        self.bundle = joblib.load(self.model_path)
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
        out["source_deltaT_available_K"] = out["T_brine_in_C"] - out["T_brine_out_min_C"]
        out["source_capacity_to_target_ratio"] = (
            4180.0
            * self.nominal_flow
            * out["source_deltaT_available_K"]
            / out["target_load_W"].clip(lower=SMALL)
        )
        out["brine_in_minus_ambient_K"] = out["T_brine_in_C"] - out["avg_T_amb_C"]
        out["brine_out_min_minus_ambient_K"] = (
            out["T_brine_out_min_C"] - out["avg_T_amb_C"]
        )
        return out[self.feature_columns]

    def domain_warnings(self, row: pd.Series) -> list[str]:
        warnings: list[str] = []
        for col, lim in self.bundle.get("input_bounds", {}).items():
            value = float(row[col])
            lo = float(lim["min"])
            hi = float(lim["max"])
            if value < lo:
                warnings.append(f"{col}={value:.6g} below training minimum {lo:.6g}")
            if value > hi:
                warnings.append(f"{col}={value:.6g} above training maximum {hi:.6g}")
        return warnings

    def _reconstruct(
        self,
        base: pd.DataFrame,
        learned: pd.DataFrame,
        fallback: np.ndarray,
        fallback_probability: np.ndarray,
    ) -> pd.DataFrame:
        out = learned.copy()
        target = base["target_load_W"].to_numpy(dtype=float)
        tamb = base["avg_T_amb_C"].to_numpy(dtype=float)
        tbrine = base["T_brine_in_C"].to_numpy(dtype=float)

        out["hp_Q_cond_W"] = np.clip(out["hp_Q_cond_W"], 0.0, target)
        out["hp_W_comp_W"] = np.clip(out["hp_W_comp_W"], SMALL, out["hp_Q_cond_W"])
        out["hp_W_comp_nominal_required_W"] = np.clip(
            out["hp_W_comp_nominal_required_W"], SMALL, target
        )
        out["hp_Q_cond_nominal_W"] = target
        out["hp_Q_cond_achievable_W"] = out["hp_Q_cond_W"]
        out["hp_Q_evap_W"] = np.maximum(out["hp_Q_cond_W"] - out["hp_W_comp_W"], 0.0)
        out["hp_Q_evap_nominal_required_W"] = np.maximum(
            target - out["hp_W_comp_nominal_required_W"], 0.0
        )
        out["hp_W_comp_operating_W"] = out["hp_W_comp_W"]
        out["hp_COP_heating"] = out["hp_Q_cond_W"] / out["hp_W_comp_W"].clip(lower=SMALL)
        out["hp_unmet_load_W"] = np.maximum(target - out["hp_Q_cond_W"], 0.0)
        out["hp_load_coverage_fraction"] = np.clip(
            out["hp_Q_cond_W"] / np.maximum(target, SMALL), 0.0, 1.0
        )
        out["hp_m_dot_ref_kg_s"] = out["hp_m_dot_ref_kg_s"].clip(lower=0.0)
        out["hp_V_s_comp_m3rev"] = float(self.constants["hp_V_s_comp_m3rev"])
        out["hp_T_brine_in_C"] = tbrine - self.hp_drop
        out["hp_T_brine_out_min_C"] = self.hp_out_min
        out["hp_T_brine_out_C"] = np.clip(
            out["hp_T_brine_out_C"], self.hp_out_min, out["hp_T_brine_in_C"]
        )
        out["hp_T_evap_C"] = out["hp_T_brine_out_C"] - 5.0
        out["hp_p_evap_bar"] = out["hp_p_evap_bar"].clip(lower=SMALL)
        out["hp_p_cond_bar"] = out["hp_p_cond_bar"].clip(lower=SMALL)
        out["hp_source_margin_at_nominal"] = out["hp_source_margin_at_nominal"].clip(lower=0.0)
        out["source_limited_nominal"] = out["hp_source_margin_at_nominal"] < 0.999

        out["orc_Q_in_W"] = out["hp_Q_evap_nominal_required_W"]
        out["orc_P_gross_W"] = out["orc_P_gross_W"].clip(lower=0.0)
        out["orc_Q_rej_W"] = out["orc_Q_rej_W"].clip(lower=0.0)
        out["orc_m_dot_orc_kg_s"] = out["orc_m_dot_orc_kg_s"].clip(lower=0.0)
        out["orc_eta_gross"] = out["orc_P_gross_W"] / out["orc_Q_in_W"].clip(lower=SMALL)
        out["orc_eta_net"] = out["orc_P_net_W"] / out["orc_Q_in_W"].clip(lower=SMALL)
        out["orc_V_s_exp_m3rev"] = np.where(
            fallback,
            float(self.constants["orc_V_s_exp_fallback_m3rev"]),
            float(self.constants["orc_V_s_exp_normal_m3rev"]),
        )
        out["orc_expander_fallback_probability"] = fallback_probability
        out["orc_T_evap_C"] = tbrine - 10.0
        out["orc_T_cond_C"] = tamb + float(self.constants["orc_cond_ambient_offset_K"])
        out["orc_design_pressure_ratio"] = (
            out["orc_p_evap_bar"].clip(lower=SMALL)
            / out["orc_p_cond_bar"].clip(lower=SMALL)
        )
        out["orc_expander_stage_count"] = np.where(
            out["orc_design_pressure_ratio"]
            > DUAL_STAGE_DESIGN_PRESSURE_RATIO_THRESHOLD,
            2.0,
            1.0,
        )

        out["common_hex_area_req_m2"] = out["common_hex_area_req_m2"].clip(lower=0.0)
        out["common_hex_orc_area_req_m2"] = np.clip(
            out["common_hex_orc_area_req_m2"], 0.0, out["common_hex_area_req_m2"]
        )
        out["common_hex_UA_required_W_K"] = (
            float(self.constants["common_hex_U_W_m2K"]) * out["common_hex_area_req_m2"]
        )
        out["common_hex_orc_fits_in_hp_common_hx"] = 1.0

        out["acc_area_req_m2"] = out["acc_area_req_m2"].clip(lower=0.0)
        out["acc_UA_required_W_K"] = (
            float(self.constants["acc_U_W_m2K"]) * out["acc_area_req_m2"]
        )
        out["acc_margin"] = out["acc_margin"].clip(lower=0.0)
        out["design_Q_geo_design_W"] = out["design_Q_geo_design_W"].clip(lower=0.0)
        out["design_m_brine_kg_s"] = self.nominal_flow

        ordered = self.report_outputs + [
            "orc_expander_fallback_probability",
            "orc_design_pressure_ratio",
            "orc_expander_stage_count",
        ]
        return out[ordered]

    def predict_frame(self, base: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.base_inputs if c not in base.columns]
        if missing:
            raise KeyError(f"Missing ANN input columns: {missing}")

        base = base[self.base_inputs].astype(float).reset_index(drop=True)
        features = self.build_features(base)
        X = self.bundle["x_scaler"].transform(features)
        pred_scaled = self.bundle["regressor"].predict(X)
        pred = self.bundle["y_scaler"].inverse_transform(pred_scaled)
        learned = pd.DataFrame(pred, columns=self.learned_outputs)

        classifier = self.bundle["expander_classifier"]
        probability = classifier.predict_proba(X)[:, 1]
        threshold = float(self.bundle["expander_classifier_threshold"])
        fallback = probability >= threshold
        return self._reconstruct(base, learned, fallback, probability)

    def predict_one(
        self,
        *,
        target_load_W: float,
        avg_T_amb_C: float,
        T_brine_in_C: float,
        p_brine_bar: float,
        T_brine_out_min_C: float,
    ) -> dict[str, Any]:
        base = pd.DataFrame([
            {
                "target_load_W": float(target_load_W),
                "avg_T_amb_C": float(avg_T_amb_C),
                "T_brine_in_C": float(T_brine_in_C),
                "p_brine_bar": float(p_brine_bar),
                "T_brine_out_min_C": float(T_brine_out_min_C),
            }
        ])
        pred = self.predict_frame(base).iloc[0]
        return {
            "inputs": {k: float(v) for k, v in base.iloc[0].to_dict().items()},
            "module_design": {
                k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                for k, v in pred.to_dict().items()
            },
            "warnings": self.domain_warnings(base.iloc[0]),
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict one nominal module with the vopt6 design ANN.")
    p.add_argument("--model", default="./models/design_ann_vopt6_module.joblib")
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
    target = args.target_load_W
    if target is None:
        target = float(args.target_load_mw) * 1.0e6
    predictor = DesignANNModule(args.model)
    result = predictor.predict_one(
        target_load_W=target,
        avg_T_amb_C=args.avg_T_amb_c,
        T_brine_in_C=args.T_brine_in_c,
        p_brine_bar=args.p_brine_bar,
        T_brine_out_min_C=args.T_brine_out_min_c,
    )
    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__":
    main()
