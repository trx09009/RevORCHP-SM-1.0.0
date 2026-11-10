# RevORCHP-SM

Surrogate models for reversible organic Rankine cycle and heat-pump systems using R1234ze(E).

## Corrected release 1.0.1

This release replaces the earlier models after correcting compressor/expander displacement sizing and independently sizing each stage of the two-stage expander. The analytical model was numerically checked, nominal hardware was recreated, every training label was regenerated, and all four surrogate bundles were retrained. The failed-search value of 204800 m³/rev is absent from the corrected hardware and learned geometry.

All nominal modules use 10 kg/s brine flow and a 1 MW HP design target. The one-stage ORC reference has mean ambient 9.7°C and design pressure ratio 6.4642; the two-stage reference uses −4°C and ratio 10.1857. Each nominal compressor and expander stage is sized at 1500 rpm. The HP nominal compressor displacement is 0.0103406 m³/rev; the colder two-stage reference has 0.0109927 m³/rev.

The design surrogate sizes variable geometry. The off-design surrogates describe their own fixed nominal hardware. An arbitrary design-surrogate geometry cannot be passed into the fixed-hardware off-design bundles. Identical modules may be aggregated by summing heat, power and mass flows; temperatures, pressures and efficiencies remain intensive.

## Install

Tested with Python 3.12.10. From this repository directory on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The thirteen tracked files are the complete runtime package. No training datasets, MATLAB files, analytical solver or CoolProp installation are needed for inference. Small property approximations/tables required for physical reconstruction are embedded in the relevant bundles and checked against the analytical property routines. In the commands below, use `.venv/Scripts/python.exe` in place of `python` when using this environment.

## Design commands

HP priority for two nominal 10 kg/s modules:

```powershell
python reversible_design_ann_vopt6_wrapper.py hp --total-heating-load-mw 1.8 --total-brine-flow-kg-s 20 --T-brine-in-hp-c 65 --avg-T-amb-c 9.7 --p-brine-bar 18 --T-brine-out-min-orc-c 70
```

ORC priority searches the supported 0.5–1.0 MW HP design interval:

```powershell
python reversible_design_ann_vopt6_wrapper.py orc --total-brine-flow-kg-s 20 --T-brine-well-c 105 --avg-T-amb-c 9.7 --p-brine-bar 18 --T-brine-out-min-orc-c 70 --grid-points 201
```

The temperature mapping is `T_brine_in_HP_C = T_brine_in_ANN_C - 40`. The design module always sizes 10 kg/s; the equal split of actual plant flow is reported separately. The HP design minimum brine outlet is 45°C. Design topology uses a strict rule: pressure ratio > 8 selects two stages; ratio ≤ 8 selects one stage.

## HP off-design command

```powershell
python hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py --model models/hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib --T-brine-in-c 65 --p-brine-bar 18 --T-brine-out-min-c 45 --m-brine-max-kg-s 10 --heating-load-mw 0.8 --temperature-lift-K 25 --sink-deltaT-K 15 --selected-only
```

The selected variant is `inverse_cop`; `direct_w` and `pressure_hybrid` remain available. Sink supply temperature is HP brine inlet plus `temperature_lift_K`; sink return is supply minus `sink_deltaT_K`. Condensation temperature is supply plus 5 K. Minimum brine outlet is a lower operating constraint. Delivered heat is capped by requested load, source heat and fixed condenser capacity, with `Q_delivered = Q_source + W_comp`. Full available brine flow is used, matching the corrected analytical controller. Raw heat/power and pressure diagnostics remain available separately from reconciled outputs. Loads below 0.2 MW are extrapolation.

## ORC off-design command

```powershell
python orc_offdesign_surrogate_filtered_variants_vopt6.py --model models/orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib --two-stage-model models/orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib --T-amb-c 9.7 --T-brine-in-c 105 --p-brine-bar 18 --T-brine-out-min-c 70 --m-brine-max-kg-s 8 --design-expander-stage-count 1
```

Use `--design-expander-stage-count 2` with the two-stage nominal hardware. Alternatively supply `--design-p-evap-bar` and `--design-p-cond-bar`; topology is determined by the **design** ratio, rather than the current operating ratio. Fixed geometry is preserved in off-design operation. Net electrical power is signed and includes generator efficiency, pump, air-condenser fan and other auxiliaries. Heat rejection includes pump shaft work. Brine source and electrical balances are reconciled explicitly.

The one-stage off-design dataset covers 3–8 kg/s of actual brine flow through the fixed 10 kg/s nominal hardware. Operation at nominal 10 kg/s is outside this surrogate's validated flow domain. The two-stage off-design dataset covers 3–10 kg/s. Nominal analytical verification and surrogate operating-domain validation are separate checks.

## Operating domain and units

Temperatures are °C, pressures are absolute bar, heat/power are W in the Python API, mass flow is kg/s, displacement is m³/rev, areas are m², and temperature differences are K. Command options ending in `mw` use MW. Input ordering and bounds are embedded in every bundle. Warnings identify marginal extrapolation; rectangular input bounds do not establish manufacturer map validity for every possible joint condition.

- **design**: `target_load_W` 500000–1e+06, `avg_T_amb_C` -10–20, `T_brine_in_C` 95–115, `p_brine_bar` 12–24, `T_brine_out_min_C` 60–75.
- **hp**: `T_brine_in_C` 55–75, `p_brine_bar` 12–24, `T_brine_out_min_C` 35–45, `m_brine_max_kg_s` 5–12, `heating_load_W` 50000–1e+06, `T_sink_supply_C` 75–100, `sink_deltaT_K` 10–25.
- **orc_one_stage**: `T_amb_C` -11.9–34.8, `T_brine_in_C` 95–115, `p_brine_bar` 12–24, `T_brine_out_min_C` 60–75, `m_brine_max_kg_s` 3–8.
- **orc_two_stage**: `T_amb_C` -15.2–34.8, `T_brine_in_C` 95–115, `p_brine_bar` 12–24, `T_brine_out_min_C` 60–75, `m_brine_max_kg_s` 3–10.

## Fresh reserved-test accuracy

Model selection used training/validation data only. Design experts retain the selected training-only fits; HP and ORC models are subsequently refitted on training+validation. Each task has 500 newly sampled test inputs frozen before label generation. HP scores cover physical requests at or above 20% nominal load, including capacity shortages. ORC scores include all physical active fresh test cases, including low or negative net power. Stress cases are separate. No cases were removed according to prediction error.

| Model | Test cases | Reported output | R² | Bundle MiB |
|---|---:|---|---:|---:|
| Design | 500 | orc_P_net_W | 0.9998 | 0.326 |
| HP off-design | 421 | hp_COP | 0.9984 | 0.299 |
| One-stage ORC off-design | 500 | orc_P_net_W | 0.9985 | 4.074 |
| Two-stage ORC off-design | 500 | orc_P_net_W | 0.9959 | 0.101 |

These scores measure agreement with the corrected thermodynamic simulator. Numerical checks do not substitute for experimental validation or establish the calibration envelope of the underlying machine maps.

## Files

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `requirements.txt`
- `bundle_io.py`
- `design_ann_vopt6_predictor.py`
- `reversible_design_ann_vopt6_wrapper.py`
- `hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py`
- `orc_offdesign_surrogate_filtered_variants_vopt6.py`
- `models/design_ann_vopt6_module.joblib`
- `models/hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib`
- `models/orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib`
- `models/orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib`

## Licence and citation

Apache-2.0; see LICENSE and CITATION.cff. Cite version 1.0.1 for these corrected models.
