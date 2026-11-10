#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design_ann_vopt6_predictor import DesignANNModule, json_default


EXTENSIVE_OUTPUTS = {
    "hp_Q_cond_W",
    "hp_Q_cond_nominal_W",
    "hp_Q_cond_achievable_W",
    "hp_Q_evap_W",
    "hp_Q_evap_nominal_required_W",
    "hp_W_comp_W",
    "hp_W_comp_nominal_required_W",
    "hp_W_comp_operating_W",
    "hp_unmet_load_W",
    "hp_m_dot_ref_kg_s",
    "hp_V_s_comp_m3rev",
    "hp_energy_balance_residual_W",
    "source_energy_residual_W",
    "orc_electrical_balance_residual_W",
    "orc_Q_in_W",
    "orc_P_gross_W",
    "orc_P_exp_el_W",
    "orc_P_pump_el_W",
    "orc_P_acc_fan_W",
    "orc_P_other_aux_W",
    "orc_P_net_W",
    "orc_Q_rej_W",
    "orc_m_dot_orc_kg_s",
    "orc_V_s_exp_m3rev",
    "orc_V_s_exp_stage1_m3rev",
    "orc_V_s_exp_stage2_m3rev",
    "common_hex_area_req_m2",
    "common_hex_UA_required_W_K",
    "common_hex_orc_area_req_m2",
    "acc_area_req_m2",
    "acc_UA_required_W_K",
    "design_Q_geo_design_W",
    "design_m_brine_kg_s",
}


def scale_installed_outputs(module: dict[str, Any], module_count: int) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for key, value in module.items():
        if isinstance(value, (bool, np.bool_)):
            total[key] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            total[key] = float(value) * module_count if key in EXTENSIVE_OUTPUTS else float(value)
        else:
            total[key] = value
    return total


class ReversibleDesignANN:
    def __init__(self, model_path: str | Path):
        self.module = DesignANNModule(model_path)
        self.nominal_flow = self.module.nominal_flow
        bounds = self.module.bundle["input_bounds"]["target_load_W"]
        self.target_min_W = float(bounds["min"])
        self.target_max_W = float(bounds["max"])

    def hp_priority(
        self,
        *,
        total_heating_load_W: float,
        total_brine_flow_kg_s: float,
        T_brine_in_HP_C: float,
        avg_T_amb_C: float,
        p_brine_bar: float,
        T_brine_out_min_ORC_C: float,
    ) -> dict[str, Any]:
        if total_heating_load_W <= 0.0:
            raise ValueError("total_heating_load_W must be positive")
        if total_brine_flow_kg_s <= 0.0:
            raise ValueError("total_brine_flow_kg_s must be positive")

        n_flow = max(1, math.ceil(total_brine_flow_kg_s / self.nominal_flow))
        n_load = max(1, math.ceil(total_heating_load_W / self.target_max_W))
        module_count = max(n_flow, n_load)
        unconstrained_target = total_heating_load_W / module_count
        module_target = float(np.clip(unconstrained_target, self.target_min_W, self.target_max_W))
        installed_nominal_load = module_count * module_target
        actual_split_flow = total_brine_flow_kg_s / module_count
        T_ann_C = T_brine_in_HP_C + self.module.hp_drop

        result = self.module.predict_one(
            target_load_W=module_target,
            avg_T_amb_C=avg_T_amb_C,
            T_brine_in_C=T_ann_C,
            p_brine_bar=p_brine_bar,
            T_brine_out_min_C=T_brine_out_min_ORC_C,
        )
        per_module = result["module_design"]
        warnings = list(result["warnings"])

        if unconstrained_target < self.target_min_W:
            warnings.append(
                "The flow-based module count makes the requested load per module lower than "
                f"the ANN minimum. Each module was sized at {self.target_min_W/1e6:.3f} MW."
            )
        if abs(actual_split_flow - self.nominal_flow) > 1.0e-9:
            warnings.append(
                "The ANN sizes a nominal 10 kg/s module. Actual performance at the equal split "
                "flow requires an off-design model for this exact sized hardware. The supplied "
                "off-design bundles describe fixed nominal 1 MW hardware."
            )

        return {
            "priority": "hp",
            "temperature_mapping": {
                "provided_T_brine_in_HP_C": float(T_brine_in_HP_C),
                "fixed_design_drop_K": float(self.module.hp_drop),
                "equivalent_T_brine_in_ANN_C": float(T_ann_C),
            },
            "modularisation": {
                "module_count": int(module_count),
                "module_count_from_flow": int(n_flow),
                "module_count_from_load": int(n_load),
                "nominal_design_flow_per_module_kg_s": float(self.nominal_flow),
                "actual_equal_split_flow_per_module_kg_s": float(actual_split_flow),
                "requested_total_heating_load_W": float(total_heating_load_W),
                "equivalent_HP_design_target_per_module_W": float(module_target),
                "installed_nominal_heating_capacity_W": float(installed_nominal_load),
            },
            "per_module_design": per_module,
            "installed_total_design": scale_installed_outputs(per_module, module_count),
            "warnings": warnings,
        }

    def orc_priority(
        self,
        *,
        total_brine_flow_kg_s: float,
        T_brine_well_C: float,
        avg_T_amb_C: float,
        p_brine_bar: float,
        T_brine_out_min_ORC_C: float,
        grid_points: int = 201,
        objective: str = "orc_P_net_W",
        min_hp_load_coverage: float = 0.99,
        require_positive_orc: bool = True,
        grid_csv_out: str | Path | None = None,
    ) -> dict[str, Any]:
        if total_brine_flow_kg_s <= 0.0:
            raise ValueError("total_brine_flow_kg_s must be positive")
        if grid_points < 2:
            raise ValueError("grid_points must be at least 2")

        module_count = max(1, math.ceil(total_brine_flow_kg_s / self.nominal_flow))
        actual_split_flow = total_brine_flow_kg_s / module_count
        targets = np.linspace(self.target_min_W, self.target_max_W, int(grid_points))
        base = pd.DataFrame(
            {
                "target_load_W": targets,
                "avg_T_amb_C": float(avg_T_amb_C),
                "T_brine_in_C": float(T_brine_well_C),
                "p_brine_bar": float(p_brine_bar),
                "T_brine_out_min_C": float(T_brine_out_min_ORC_C),
            }
        )
        pred = self.module.predict_frame(base)
        if objective not in pred.columns:
            raise KeyError(f"Unknown ORC objective {objective!r}")

        feasible = np.isfinite(pred[objective].to_numpy(dtype=float))
        feasible &= pred["hp_load_coverage_fraction"].to_numpy(dtype=float) >= float(min_hp_load_coverage)
        if require_positive_orc:
            feasible &= pred[objective].to_numpy(dtype=float) > 0.0

        search = base.copy()
        search["feasible"] = feasible
        for col in [
            objective,
            "orc_P_gross_W",
            "orc_Q_in_W",
            "orc_eta_net",
            "hp_Q_cond_W",
            "hp_load_coverage_fraction",
            "hp_source_margin_at_nominal",
            "common_hex_area_req_m2",
            "acc_area_req_m2",
        ]:
            if col in pred.columns:
                search[col] = pred[col].to_numpy()
        search[f"installed_total_{objective}"] = module_count * search[objective]

        if grid_csv_out is not None:
            out_path = Path(grid_csv_out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            search.to_csv(out_path, index=False)

        valid_idx = np.flatnonzero(feasible)
        if len(valid_idx) == 0:
            raise RuntimeError(
                "No feasible ORC-priority candidate was found inside the ANN target range. "
                "Reduce min_hp_load_coverage or inspect the source conditions."
            )

        objective_values = pred[objective].to_numpy(dtype=float)
        best_idx = int(valid_idx[np.argmax(objective_values[valid_idx])])
        selected = pred.iloc[best_idx].to_dict()
        selected = {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
            for k, v in selected.items()
        }
        boundary = best_idx == 0 or best_idx == len(targets) - 1
        warnings = self.module.domain_warnings(base.iloc[best_idx])

        if boundary:
            edge = "lower" if best_idx == 0 else "upper"
            warnings.append(
                f"The ORC optimum lies at the {edge} ANN target boundary. It is the best "
                "design inside the existing 0.5–1.0 MW training interval, not proof of a global optimum."
            )
        if abs(actual_split_flow - self.nominal_flow) > 1.0e-9:
            warnings.append(
                "The installed design is based on nominal 10 kg/s modules. Actual ORC output at "
                "the equal split flow requires an off-design model for this exact sized hardware. "
                "The supplied off-design bundles describe fixed nominal 1 MW hardware."
            )

        ranked = search.loc[feasible].nlargest(min(10, int(feasible.sum())), objective)
        top_candidates = ranked[
            [
                "target_load_W",
                objective,
                f"installed_total_{objective}",
                "orc_Q_in_W",
                "orc_eta_net",
                "hp_load_coverage_fraction",
            ]
        ].to_dict(orient="records")

        return {
            "priority": "orc",
            "temperature_mapping": {
                "provided_T_brine_well_C": float(T_brine_well_C),
                "equivalent_T_brine_in_ANN_C": float(T_brine_well_C),
                "fixed_design_drop_K": float(self.module.hp_drop),
                "implied_T_brine_in_HP_C": float(T_brine_well_C - self.module.hp_drop),
            },
            "modularisation": {
                "module_count": int(module_count),
                "total_brine_flow_kg_s": float(total_brine_flow_kg_s),
                "nominal_design_flow_per_module_kg_s": float(self.nominal_flow),
                "actual_equal_split_flow_per_module_kg_s": float(actual_split_flow),
            },
            "inverse_search": {
                "objective": objective,
                "grid_points": int(grid_points),
                "target_min_W": float(self.target_min_W),
                "target_max_W": float(self.target_max_W),
                "min_hp_load_coverage": float(min_hp_load_coverage),
                "selected_equivalent_HP_design_target_W": float(targets[best_idx]),
                "selected_objective_per_module_W": float(objective_values[best_idx]),
                "selected_objective_installed_total_W": float(
                    module_count * objective_values[best_idx]
                ),
                "selected_at_training_boundary": bool(boundary),
                "top_candidates": top_candidates,
            },
            "per_module_design": selected,
            "installed_total_design": scale_installed_outputs(selected, module_count),
            "warnings": warnings,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HP- and ORC-priority wrapper for the design ANN.")
    parser.add_argument("--model", default="./models/design_ann_vopt6_module.joblib")
    sub = parser.add_subparsers(dest="priority", required=True)

    hp = sub.add_parser("hp", help="HP-priority design")
    hp.add_argument("--total-heating-load-mw", type=float, required=True)
    hp.add_argument("--total-brine-flow-kg-s", type=float, required=True)
    hp.add_argument("--T-brine-in-hp-c", type=float, required=True)
    hp.add_argument("--avg-T-amb-c", type=float, required=True)
    hp.add_argument("--p-brine-bar", type=float, required=True)
    hp.add_argument("--T-brine-out-min-orc-c", type=float, required=True)

    orc = sub.add_parser("orc", help="ORC-priority inverse design")
    orc.add_argument("--total-brine-flow-kg-s", type=float, required=True)
    orc.add_argument("--T-brine-well-c", type=float, required=True)
    orc.add_argument("--avg-T-amb-c", type=float, required=True)
    orc.add_argument("--p-brine-bar", type=float, required=True)
    orc.add_argument("--T-brine-out-min-orc-c", type=float, required=True)
    orc.add_argument("--grid-points", type=int, default=201)
    orc.add_argument("--objective", choices=["orc_P_net_W", "orc_P_gross_W"], default="orc_P_net_W")
    orc.add_argument("--min-hp-load-coverage", type=float, default=0.99)
    orc.add_argument("--allow-nonpositive-orc", action="store_true")
    orc.add_argument("--grid-csv-out", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    wrapper = ReversibleDesignANN(args.model)

    if args.priority == "hp":
        result = wrapper.hp_priority(
            total_heating_load_W=float(args.total_heating_load_mw) * 1.0e6,
            total_brine_flow_kg_s=args.total_brine_flow_kg_s,
            T_brine_in_HP_C=args.T_brine_in_hp_c,
            avg_T_amb_C=args.avg_T_amb_c,
            p_brine_bar=args.p_brine_bar,
            T_brine_out_min_ORC_C=args.T_brine_out_min_orc_c,
        )
    else:
        result = wrapper.orc_priority(
            total_brine_flow_kg_s=args.total_brine_flow_kg_s,
            T_brine_well_C=args.T_brine_well_c,
            avg_T_amb_C=args.avg_T_amb_c,
            p_brine_bar=args.p_brine_bar,
            T_brine_out_min_ORC_C=args.T_brine_out_min_orc_c,
            grid_points=args.grid_points,
            objective=args.objective,
            min_hp_load_coverage=args.min_hp_load_coverage,
            require_positive_orc=not args.allow_nonpositive_orc,
            grid_csv_out=args.grid_csv_out,
        )

    print(json.dumps(result, indent=2, default=json_default))


if __name__ == "__main__":
    main()
