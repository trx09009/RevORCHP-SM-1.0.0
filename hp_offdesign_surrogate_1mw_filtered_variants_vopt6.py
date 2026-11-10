#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from bundle_io import load_bundle, predict_model


SMALL = 1.0e-12


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


def undo_log(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y, dtype=float))


def eval_eta_poly(poly: dict[str, Any], pr: np.ndarray) -> np.ndarray:
    eta = np.polyval(np.asarray(poly["coeffs_desc"], dtype=float), np.asarray(pr, dtype=float))
    return np.clip(eta, float(poly["eta_clip_min"]), float(poly["eta_clip_max"]))


def build_feature_frame(
    *,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    heating_load_W: float,
    temperature_lift_K: float,
    sink_deltaT_K: float,
    sink_approach_K: float,
    nominal_load_W: float,
    cp_proxy_J_kgK: float,
    input_columns: list[str],
) -> pd.DataFrame:
    row: dict[str, float] = {
        "T_brine_in_C": float(T_brine_in_C),
        "p_brine_bar": float(p_brine_bar),
        "T_brine_out_min_C": float(T_brine_out_min_C),
        "m_brine_max_kg_s": float(m_brine_max_kg_s),
        "heating_load_W": float(heating_load_W),
        "temperature_lift_K": float(temperature_lift_K),
        "sink_deltaT_K": float(sink_deltaT_K),
    }

    row["T_sink_supply_C"] = row["T_brine_in_C"] + row["temperature_lift_K"]
    row["T_sink_return_C"] = row["T_sink_supply_C"] - row["sink_deltaT_K"]
    row["T_cond_HP_target_C"] = row["T_sink_supply_C"] + float(sink_approach_K)
    row["load_fraction_nominal"] = row["heating_load_W"] / max(float(nominal_load_W), 1.0)
    row["source_deltaT_available_K"] = row["T_brine_in_C"] - row["T_brine_out_min_C"]
    row["geo_capacity_proxy_W"] = (
        row["m_brine_max_kg_s"] * float(cp_proxy_J_kgK) * row["source_deltaT_available_K"]
    )
    row["sink_supply_minus_brine_out_min_K"] = row["T_sink_supply_C"] - row["T_brine_out_min_C"]
    row["cond_target_minus_brine_in_K"] = row["T_cond_HP_target_C"] - row["T_brine_in_C"]
    row["cond_target_minus_brine_out_min_K"] = row["T_cond_HP_target_C"] - row["T_brine_out_min_C"]

    if abs(row["source_deltaT_available_K"]) > SMALL:
        row["sink_deltaT_over_source_deltaT"] = row["sink_deltaT_K"] / row["source_deltaT_available_K"]
    else:
        row["sink_deltaT_over_source_deltaT"] = np.nan

    missing = [c for c in input_columns if c not in row]
    if missing:
        raise KeyError(f"Missing HP input columns: {missing}")

    return pd.DataFrame([{c: row[c] for c in input_columns}])


def domain_warnings(row: pd.Series, bounds: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for col, lim in (bounds or {}).items():
        if col not in row.index:
            continue
        val = float(row[col])
        lo = lim.get("min")
        hi = lim.get("max")
        if lo is not None and val < float(lo):
            warnings.append(f"{col}={val:.6g} below training range min {float(lo):.6g}")
        if hi is not None and val > float(hi):
            warnings.append(f"{col}={val:.6g} above training range max {float(hi):.6g}")
    return warnings


def compute_variant(bundle: dict[str, Any], X: np.ndarray, q_pred: np.ndarray, variant: str) -> dict[str, float]:
    if variant not in bundle["variant_models"]:
        raise KeyError(
            f"Variant {variant!r} is not available. "
            f"Available variants: {list(bundle['variant_models'].keys())}"
        )

    q_cols = list(bundle["q_output_columns"])
    cfg = bundle["variant_configs"][variant]
    cols = cfg.get("cols") or cfg.get("target_columns")
    if not cols:
        raise KeyError(f"No target columns found for HP variant {variant!r}")

    log_cols = set(cfg.get("log_cols", [])) | set(cfg.get("log_columns", []))
    raw = np.asarray(predict_model(bundle["variant_models"][variant],X), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    out: dict[str, float] = {}
    for i, col in enumerate(q_cols):
        out[col] = float(q_pred[0, i])

    for j, col in enumerate(cols):
        val = raw[0, j]
        if col in log_cols:
            val = undo_log(np.array([val]))[0]
        out[col] = float(val)

    Q = max(float(out["hp_Q_delivered_W"]), 0.0)
    source_diagnostic = max(float(out.get("hp_Q_source_W", 0.0)), 0.0)

    if variant == "direct_w":
        W = max(float(out["hp_W_comp_W"]), SMALL)
        out["hp_inverse_COP"] = W / max(Q, SMALL)
        out["hp_COP"] = Q / W

    elif variant == "inverse_cop":
        inv = max(float(out["hp_inverse_COP"]), SMALL)
        W = Q * inv
        out["hp_W_comp_W"] = W
        out["hp_COP"] = 1.0 / inv

    elif variant == "pressure_hybrid":
        pcond = max(float(out["hp_p_cond_bar"]), SMALL)
        pevap = max(float(out["hp_p_evap_bar"]), SMALL)
        pr = pcond / pevap
        # Corrected models learn the actual map efficiency. Historical bundles
        # used a pressure-only polynomial, which cannot express flow effects.
        eta = (float(np.clip(out["hp_eta_is"], 0.05, 1.0))
               if "hp_eta_is" in out else float(eval_eta_poly(bundle["eta_poly"], np.array([pr]))[0]))
        Wis = max(float(out["hp_W_is_proxy_W"]), SMALL)
        W = Wis / max(eta, SMALL)
        out["hp_pressure_ratio"] = pr
        out["hp_eta_is_fit"] = eta
        out["hp_W_comp_W"] = W
        out["hp_inverse_COP"] = W / max(Q, SMALL)
        out["hp_COP"] = Q / max(W, SMALL)

    else:
        raise ValueError(f"Unknown HP variant: {variant}")

    W = float(np.clip(out["hp_W_comp_W"], 0.0, Q))
    out["hp_Q_delivered_W"] = Q
    out["hp_W_comp_W"] = W
    out["hp_inverse_COP"] = W / Q if Q > 0 else 0.0
    out["hp_COP"] = Q / max(W, SMALL) if Q > 0 else 0.0
    out["hp_Q_source_W"] = Q - W
    out["hp_Q_source_surrogate_W"] = source_diagnostic
    out["hp_energy_balance_residual_W"] = Q - out["hp_Q_source_W"] - W
    return out


def enforce_operating_constraints(out, *, heating_load_W, T_brine_in_C,
                                  T_brine_out_min_C, m_brine_max_kg_s,
                                  cp_J_kgK=4200.0, full_available_flow=False,
                                  UA_common_W_K=None, sink_deltaT_K=None,
                                  sink_approach_K=5.0):
    """Project onto the existing constant-cp source envelope at predicted COP.

    This protects feasibility outside stable fitting data; it is not a detailed
    thermodynamic re-solve. The unprojected source estimate remains diagnostic.
    """
    out = dict(out)
    q = max(float(out["hp_Q_delivered_W"]), 0.0)
    w = float(np.clip(out["hp_W_comp_W"], 0.0, q))
    source = q - w
    delta = max(T_brine_in_C - T_brine_out_min_C, 0.0)
    capacity = cp_J_kgK * m_brine_max_kg_s * delta
    hx_limit = (float(UA_common_W_K) * sink_deltaT_K /
                np.log1p(sink_deltaT_K / sink_approach_K)
                if UA_common_W_K is not None and sink_deltaT_K is not None else float("inf"))
    factor = min(1.0, heating_load_W / q if q > 0 else 1.0,
                 capacity / source if source > 0 else 1.0,
                 hx_limit / q if q > 0 else 1.0)
    out["hp_constraint_projection_applied"] = bool(factor < 1.0 - 1e-12)
    out["hp_Q_delivered_W"] = q * factor
    out["hp_W_comp_W"] = w * factor
    out["hp_Q_source_W"] = source * factor
    q, w, source = (out[k] for k in ("hp_Q_delivered_W", "hp_W_comp_W", "hp_Q_source_W"))
    out["hp_inverse_COP"] = w / q if q > 0 else 0.0
    out["hp_COP"] = q / max(w, SMALL) if q > 0 else 0.0
    required_mass = source / (cp_J_kgK * delta) if delta > 0 else 0.0
    mass = float(np.clip(out.get("hp_m_brine_kg_s", m_brine_max_kg_s),
                         required_mass, m_brine_max_kg_s))
    if full_available_flow:
        mass = float(m_brine_max_kg_s)
    out["hp_m_brine_kg_s"] = mass
    out["hp_T_brine_out_C"] = T_brine_in_C - source / (cp_J_kgK * mass) if mass > 0 else T_brine_in_C
    if "hp_T_evap_C" in out:
        out["hp_T_evap_C"] = out["hp_T_brine_out_C"] - 5.0
    out["hp_unmet_load_W"] = max(heating_load_W - q, 0.0)
    out["hp_load_coverage_frac"] = q / heating_load_W if heating_load_W > 0 else 0.0
    out["hp_source_capacity_limit_W"] = capacity
    if np.isfinite(hx_limit): out["hp_condenser_capacity_limit_W"] = hx_limit
    out["hp_T_evap_C"] = out["hp_T_brine_out_C"] - 5.0
    out["hp_energy_balance_residual_W"] = q - source - w
    return out


def reconcile_pressures(out, bundle, *, T_brine_in_C, temperature_lift_K):
    """Reconstruct saturation pressure at final temperatures without CoolProp.

    Learned pressure-head values remain diagnostics. Outside the embedded
    saturation range actual pressure fields are omitted, never endpoint-clipped.
    """
    out = dict(out)
    table = bundle.get("saturation_pressure_table")
    if table is None:
        return out
    for field in ["hp_p_cond_bar", "hp_p_evap_bar"]:
        if field in out:
            out[field.replace("_bar", "_surrogate_bar")] = out.pop(field)
    out["hp_T_cond_C"] = T_brine_in_C + temperature_lift_K + float(bundle.get("sink_approach_K", 5.))
    temps = np.asarray(table["temperature_C"], dtype=float)
    pressures = np.asarray(table["pressure_bar"], dtype=float)
    supported = True
    for target, temperature in [("hp_p_cond_bar", out["hp_T_cond_C"]),
                                ("hp_p_evap_bar", out["hp_T_evap_C"])]:
        if temps[0] <= temperature <= temps[-1]:
            out[target] = float(np.interp(temperature, temps, pressures))
        else:
            supported = False
    out["hp_pressure_state_supported"] = supported
    out.pop("hp_pressure_ratio", None)
    if supported:
        out["hp_pressure_ratio"] = out["hp_p_cond_bar"] / out["hp_p_evap_bar"]
    return out


def predict_1mw_hp_performance(
    *,
    model_path: str | Path,
    T_brine_in_C: float,
    p_brine_bar: float,
    T_brine_out_min_C: float,
    m_brine_max_kg_s: float,
    heating_load_W: float,
    temperature_lift_K: float,
    sink_deltaT_K: float,
    variant: str | None = None,
    return_all_variants: bool = True,
) -> dict[str, Any]:
    values = [T_brine_in_C, p_brine_bar, T_brine_out_min_C,
              m_brine_max_kg_s, heating_load_W, temperature_lift_K, sink_deltaT_K]
    if not np.isfinite(values).all():
        raise ValueError("HP inputs must be finite")
    if (T_brine_in_C <= T_brine_out_min_C or p_brine_bar <= 0
            or m_brine_max_kg_s <= 0 or heating_load_W < 0 or sink_deltaT_K <= 0):
        raise ValueError("Invalid HP operating boundary, flow, pressure, load or sink difference")
    bundle = load_bundle(model_path)
    input_columns = list(bundle["input_columns"])

    Xdf = build_feature_frame(
        T_brine_in_C=T_brine_in_C,
        p_brine_bar=p_brine_bar,
        T_brine_out_min_C=T_brine_out_min_C,
        m_brine_max_kg_s=m_brine_max_kg_s,
        heating_load_W=heating_load_W,
        temperature_lift_K=temperature_lift_K,
        sink_deltaT_K=sink_deltaT_K,
        sink_approach_K=float(bundle.get("sink_approach_K", 5.0)),
        nominal_load_W=float(bundle.get("nominal_load_W", 1.0e6)),
        cp_proxy_J_kgK=float(bundle.get("cp_proxy_J_kgK", 4180.0)),
        input_columns=input_columns,
    )

    X = Xdf[input_columns].to_numpy(dtype=float)
    q_pred = np.asarray(predict_model(bundle["q_model"],X), dtype=float)
    if q_pred.ndim == 1:
        q_pred = q_pred.reshape(1, -1)

    selected = variant or bundle.get("selected_variant", "inverse_cop")
    classifier = bundle.get("physical_status_classifier")
    status_supported = True if classifier is None else bool(predict_model(classifier, X)[0])
    def evaluate(name):
        out = compute_variant(bundle, X, q_pred, name)
        out["hp_Q_delivered_raw_W"] = out["hp_Q_delivered_W"]
        out["hp_W_comp_raw_W"] = out["hp_W_comp_W"]
        if not status_supported:
            out["hp_Q_delivered_W"] = out["hp_W_comp_W"] = 0.0
        out = enforce_operating_constraints(
            out, heating_load_W=heating_load_W,
            T_brine_in_C=T_brine_in_C, T_brine_out_min_C=T_brine_out_min_C,
            m_brine_max_kg_s=m_brine_max_kg_s,
            cp_J_kgK=float(bundle.get("source_cp_J_kgK", 4200.0)),
            full_available_flow=bundle.get("source_flow_policy") == "full_available",
            UA_common_W_K=bundle.get("reference_hardware", {}).get("UA_common_W_K"),
            sink_deltaT_K=sink_deltaT_K, sink_approach_K=float(bundle.get("sink_approach_K", 5.)))
        out["hp_physical_status_supported"] = status_supported
        return reconcile_pressures(out, bundle, T_brine_in_C=T_brine_in_C,
                                   temperature_lift_K=temperature_lift_K)

    result: dict[str, Any] = {
        "surrogate": "hp_offdesign",
        "selected_variant": selected,
        "inputs": {k: float(v) for k, v in Xdf.iloc[0].to_dict().items()},
        "hp": evaluate(selected),
        "warnings": domain_warnings(Xdf.iloc[0], bundle.get("input_bounds_filtered", {})),
        "model_metadata": {k: bundle.get(k) for k in ["bundle_version", "reference_source_sha256", "reference_hardware_sha256"]},
    }

    if return_all_variants:
        result["variants"] = {
            name: (result["hp"] if name == selected else evaluate(name))
            for name in bundle["variant_models"].keys()
        }

    if result["hp"]["hp_constraint_projection_applied"]:
        result["warnings"].append("HP output projected to requested-load, condenser and source-capacity limits at predictedCOP.")
    if heating_load_W < float(bundle.get("minimum_fit_load_W", 200000.)):
        result["warnings"].append("Requested load is below the20% fitting regime; this is extrapolation.")
    if not status_supported:
        result["warnings"].append("Input is classified outside the supported active physical regime; heat and power are returned as zero.")
    if result["hp"].get("hp_pressure_state_supported") is False:
        result["warnings"].append("Final temperature is outside the supported saturation-pressure range; unsupported actual pressure fields are omitted. No endpoint clipping is applied.")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict 1 MW HP off-design performance.")
    p.add_argument("--model", required=True)
    p.add_argument("--T-brine-in-c", type=float, required=True)
    p.add_argument("--p-brine-bar", type=float, required=True)
    p.add_argument("--T-brine-out-min-c", type=float, required=True)
    p.add_argument("--m-brine-max-kg-s", type=float, required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--heating-load-W", type=float)
    group.add_argument("--heating-load-mw", type=float)

    p.add_argument("--temperature-lift-K", type=float, required=True)
    p.add_argument("--sink-deltaT-K", type=float, required=True)
    p.add_argument("--variant", choices=["direct_w", "inverse_cop", "pressure_hybrid"], default=None)
    p.add_argument("--selected-only", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    heating_load_W = args.heating_load_W
    if heating_load_W is None:
        heating_load_W = float(args.heating_load_mw) * 1.0e6

    result = predict_1mw_hp_performance(
        model_path=args.model,
        T_brine_in_C=args.T_brine_in_c,
        p_brine_bar=args.p_brine_bar,
        T_brine_out_min_C=args.T_brine_out_min_c,
        m_brine_max_kg_s=args.m_brine_max_kg_s,
        heating_load_W=heating_load_W,
        temperature_lift_K=args.temperature_lift_K,
        sink_deltaT_K=args.sink_deltaT_K,
        variant=args.variant,
        return_all_variants=not bool(args.selected_only),
    )
    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__":
    main()
