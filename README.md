# RevORCHP-SM

**Surrogate models for reversible organic Rankine cycle/heat pump systems**

| Release | Date |
| --- | --- |
| 1.0.0 | 10 November 2026 |

RevORCHP-SM provides Python prediction wrappers for the design and off-design performance of reversible organic Rankine cycle and heat pump systems. The package includes a modular design ANN, ORC off-design surrogates for one-stage and two-stage expander configurations, and a source-balanced HP off-design surrogate.

The current models use temperature ranges relevant to geothermal applications. Their applicability depends on the training domain and nominal configuration of each released model bundle.

The software was developed within the FLEXGEO project.

## Citation

If you use these surrogate models, prediction scripts, or their outputs in a publication, report, or other research work, please cite the software version used.

Roumpedakis, T. (2026). *RevORCHP-SM — Surrogate models for reversible organic Rankine cycle/heat pump systems* (Version 1.0.0) [Computer software]. https://github.com/trx09009/RevORCHP-SM-1.0.0

Citation metadata, including the author's ORCID and affiliation, is provided in `CITATION.cff`. Use **Cite this repository** on GitHub to obtain APA and BibTeX references.

## Package contents

Keep the following relative paths when assembling or downloading the release. Run the examples from the repository's main folder.

| File | Purpose |
| --- | --- |
| `README.md` | Model documentation and usage instructions |
| `CITATION.cff` | Software citation metadata |
| `LICENSE` | Apache License 2.0 terms |
| `reversible_design_ann_vopt6_wrapper.py` | Modular design wrapper |
| `design_ann_vopt6_predictor.py` | Design ANN prediction module used by the design wrapper |
| `orc_offdesign_surrogate_filtered_variants_vopt6.py` | ORC off-design prediction wrapper |
| `hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py` | HP off-design prediction wrapper |
| `models/design_ann_vopt6_module.joblib` | Design ANN bundle |
| `models/orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib` | One-stage ORC off-design bundle |
| `models/orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib` | Two-stage ORC off-design bundle |
| `models/hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib` | Source-balanced HP off-design bundle |

## Installation

The wrapper documentation specifies Python 3.10 or newer and the following packages.

- NumPy
- pandas
- joblib
- scikit-learn

The commands below use Windows PowerShell and a local virtual environment. Open PowerShell in the repository's main folder and run the following commands.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas joblib scikit-learn
```

The installation command uses unpinned package versions. Loading the serialized models reproducibly requires the Python and dependency versions used to create the released bundles. Exact tested versions remain to be recorded for this release. Scikit-learn recommends matching the training environment when loading persisted estimators. [Model persistence documentation](https://scikit-learn.org/stable/model_persistence.html)

## Inputs, units and operating scope

Temperatures are supplied in degrees Celsius, temperature differences in kelvin, brine pressure in bar, and mass flow in kg/s. Heating requests use MW unless the argument explicitly specifies watts. Output names ending in `_W` use watts.

| Model | Operating scope documented for this release |
| --- | --- |
| Design ANN | Nominal design module at 10 kg/s. Complete numerical input bounds remain to be documented |
| ORC off-design | One module, with one-stage or two-stage topology selected from the design. Complete numerical input bounds remain to be documented |
| HP off-design | One 1 MW-class module, with the operating ranges listed below |

### HP operating ranges

| Quantity | Recommended range |
| --- | --- |
| HP brine inlet temperature | 55–75 °C |
| Minimum permitted HP brine outlet temperature | 35–45 °C |
| Brine pressure | 12–24 bar |
| Available brine mass flow per module | 5–12 kg/s |
| Heating load used for stable fitting, per module | Approximately 0.2–1.0 MW |
| Sink supply temperature | 75–100 °C |
| Sink supply-to-return temperature difference | 10–25 K |

These are marginal operating ranges. Coverage of combined input conditions depends on the joint training distribution. Inspect the training-domain warnings returned by the wrappers and review extrapolated predictions against the detailed model.

## Design model

The design ANN represents a nominal module designed at a brine flow of 10 kg/s. For larger total brine flows, the wrapper selects an integer number of modules and distributes the available flow equally between them.

The design wrapper uses the following temperature mapping.

```text
T_brine_in_HP_C = T_brine_in_ANN_C - 40
```

The offset is 40 K. For example, an HP brine inlet temperature of 65 °C maps to a design ANN brine input of 105 °C. The wrapper returns the temperature mapping and warnings about training boundaries and differences between nominal and actual module flow.

### Design from a total heating requirement

| Argument | Meaning | Unit |
| --- | --- | --- |
| `--total-heating-load-mw` | Required total heating capacity | MW |
| `--total-brine-flow-kg-s` | Total available brine mass flow | kg/s |
| `--T-brine-in-hp-c` | Brine temperature at the HP evaporator inlet | °C |
| `--avg-T-amb-c` | Average ambient temperature | °C |
| `--p-brine-bar` | Brine pressure | bar |
| `--T-brine-out-min-orc-c` | Minimum ORC-side brine outlet temperature | °C |

```powershell
.\.venv\Scripts\python.exe .\reversible_design_ann_vopt6_wrapper.py hp `
  --total-heating-load-mw 2.2 `
  --total-brine-flow-kg-s 25 `
  --T-brine-in-hp-c 65 `
  --avg-T-amb-c 5 `
  --p-brine-bar 18 `
  --T-brine-out-min-orc-c 70
```

### Design from the well condition

| Argument | Meaning | Unit or default |
| --- | --- | --- |
| `--total-brine-flow-kg-s` | Total available brine mass flow | kg/s |
| `--T-brine-well-c` | Well/ORC brine inlet temperature | °C |
| `--avg-T-amb-c` | Average ambient temperature | °C |
| `--p-brine-bar` | Brine pressure | bar |
| `--T-brine-out-min-orc-c` | Minimum ORC-side brine outlet temperature | °C |
| `--grid-points` | Number of candidate module design targets evaluated | 201 |
| `--grid-csv-out` | Optional path for saving the evaluated candidate grid | File path |

```powershell
.\.venv\Scripts\python.exe .\reversible_design_ann_vopt6_wrapper.py orc `
  --total-brine-flow-kg-s 25 `
  --T-brine-well-c 106 `
  --avg-T-amb-c 5 `
  --p-brine-bar 18 `
  --T-brine-out-min-orc-c 70 `
  --grid-points 201
```

### Design outputs

| Output group | Content |
| --- | --- |
| `temperature_mapping` | Supplied and implied temperatures under the design convention |
| `modularisation` | Module count, nominal module flow, actual equal flow split and module design target |
| `per_module_design` | HP, ORC, heat-exchanger, air-cooled condenser and source design for one module |
| `installed_total_design` | Installation-level extensive quantities scaled by the module count |
| `inverse_search` | Candidate-search information for the well-condition command |
| `warnings` | Training-boundary and nominal-versus-actual-flow warnings |

The per-module design includes `orc_design_pressure_ratio` and `orc_expander_stage_count`. Pass the stage count or the design evaporation and condensation pressures to the ORC off-design wrapper.

## ORC off-design model

Each call evaluates one module. The wrapper supports one-stage and two-stage expander bundles.

### Inputs

| Argument | Meaning | Unit |
| --- | --- | --- |
| `--T-amb-c` | Current ambient temperature | °C |
| `--T-brine-in-c` | ORC brine inlet temperature | °C |
| `--p-brine-bar` | Brine pressure | bar |
| `--T-brine-out-min-c` | Minimum permitted ORC brine outlet temperature | °C |
| `--m-brine-max-kg-s` | Available brine flow for one module | kg/s |
| `--variant` | Optional model variant. The bundle's stored selection is used when omitted | Variant name |
| `--design-expander-stage-count` | Explicit design topology | 1 or 2 |
| `--design-p-evap-bar` | Design evaporation pressure | bar |
| `--design-p-cond-bar` | Design condensation pressure | bar |

Specify the topology through either `--design-expander-stage-count` or the pair of design pressures. With the pressure-based selection, the wrapper calculates the design pressure ratio.

```text
design_pressure_ratio = design_p_evap_bar / design_p_cond_bar
```

| Design pressure ratio | Selected topology |
| --- | --- |
| Greater than 8.0 | Two-stage expander |
| Up to and including 8.0 | One-stage expander |

The documented default is the legacy one-stage model when topology information is omitted.

### One-stage example

The design pressure ratio in this example is 28/4 = 7.

```powershell
.\.venv\Scripts\python.exe .\orc_offdesign_surrogate_filtered_variants_vopt6.py `
  --model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib `
  --two-stage-model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib `
  --T-amb-c 15 `
  --T-brine-in-c 106 `
  --p-brine-bar 18 `
  --T-brine-out-min-c 70 `
  --m-brine-max-kg-s 8.33 `
  --design-p-evap-bar 28 `
  --design-p-cond-bar 4
```

### Two-stage example

The design pressure ratio in this example is 28/3, approximately 9.33.

```powershell
.\.venv\Scripts\python.exe .\orc_offdesign_surrogate_filtered_variants_vopt6.py `
  --model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6.joblib `
  --two-stage-model .\models\orc_synthetic_year_surrogate_filtered_variants_vopt6_two_stage.joblib `
  --T-amb-c 0 `
  --T-brine-in-c 106 `
  --p-brine-bar 18 `
  --T-brine-out-min-c 70 `
  --m-brine-max-kg-s 8.33 `
  --design-p-evap-bar 28 `
  --design-p-cond-bar 3
```

### ORC outputs

| Output | Meaning | Unit |
| --- | --- | --- |
| `orc_Q_in_W` | ORC heat input | W |
| `orc_Q_rej_W` | Rejected heat | W |
| `orc_P_gross_W` | Gross ORC power | W |
| `orc_P_net_W` | Net ORC power | W |
| `orc_eta_gross` | Gross efficiency | Fraction |
| `orc_eta_net` | Net efficiency | Fraction |
| `orc_P_pump_el_W` | Pump electrical consumption | W |
| `orc_P_acc_fan_W` | Air-cooled condenser fan consumption | W |
| `orc_P_other_aux_W` | Other auxiliary consumption | W |
| `orc_m_brine_kg_s` | Predicted brine mass flow | kg/s |
| `orc_T_brine_out_C` | Predicted brine outlet temperature | °C |
| `orc_Q_geo_avail_W` | Available geothermal heat | W |
| `orc_Q_geo_used_W` | Used geothermal heat | W |

The `expander_selection` output reports the selected stage count, design pressure ratio, threshold and model path. The response also includes the selected variant, available variant results, derived inputs and training-domain warnings.

## HP off-design model

Each call evaluates one module using the source-balanced HP bundle listed in the package contents.

### Inputs

| Argument | Meaning | Unit |
| --- | --- | --- |
| `--T-brine-in-c` | Brine temperature at the HP evaporator inlet | °C |
| `--p-brine-bar` | Brine pressure | bar |
| `--T-brine-out-min-c` | Lower permitted HP brine outlet temperature | °C |
| `--m-brine-max-kg-s` | Available brine mass flow for one module | kg/s |
| `--heating-load-mw` | Requested heating load for one module | MW |
| `--heating-load-W` | Alternative argument for the same heating request in watts | W |
| `--temperature-lift-K` | Sink supply temperature minus HP brine inlet temperature | K |
| `--sink-deltaT-K` | Sink supply temperature minus sink return temperature | K |
| `--variant` | Optional variant. The bundle's selected inverse-COP variant is used when omitted | Variant name |
| `--selected-only` | Restricts the reported results to the selected variant | Flag |

Use one heating-load argument for each call. The minimum brine outlet temperature is a lower operating constraint. With sufficient source heat, the model can predict a warmer outlet and a higher evaporation temperature.

```text
temperature_lift_K = T_sink_supply_C - T_brine_in_C
sink_deltaT_K = T_sink_supply_C - T_sink_return_C
```

### Example

The inputs below correspond to a sink supply temperature of 90 °C and a sink return temperature of 75 °C.

```powershell
.\.venv\Scripts\python.exe .\hp_offdesign_surrogate_1mw_filtered_variants_vopt6.py `
  --model .\models\hp_synthetic_year_surrogate_1mw_filtered_variants_vopt6_hp55_75_out35_45_sourcebalanced.joblib `
  --T-brine-in-c 65 `
  --p-brine-bar 18 `
  --T-brine-out-min-c 40 `
  --m-brine-max-kg-s 8.33 `
  --heating-load-mw 0.8 `
  --temperature-lift-K 25 `
  --sink-deltaT-K 15 `
  --selected-only
```

### HP outputs

| Output | Meaning | Unit |
| --- | --- | --- |
| `hp_Q_delivered_W` | Predicted useful heat delivered by the HP | W |
| `hp_inverse_COP` | Predicted inverse COP | Dimensionless |
| `hp_COP` | Reciprocal of `hp_inverse_COP` | Dimensionless |
| `hp_W_comp_W` | Delivered heat multiplied by inverse COP | W |
| `hp_Q_source_W` | Delivered heat minus compressor power | W |
| `hp_Q_source_surrogate_W` | Independent source-heat estimate for diagnostic comparison | W |
| `hp_energy_balance_residual_W` | Residual of the enforced energy balance | W |
| `hp_unmet_load_W` | Unmet heating request | W |
| `hp_load_coverage_frac` | Fraction of the requested heating load covered | Fraction |
| `hp_m_brine_kg_s` | Predicted brine mass flow | kg/s |
| `hp_T_brine_out_C` | Predicted brine outlet temperature | °C |

The wrapper enforces the following balance.

```text
hp_Q_delivered_W = hp_Q_source_W + hp_W_comp_W
```

The balance residual should approach zero within numerical precision. This residual checks algebraic consistency. Prediction accuracy requires separate validation against reference results.

## Integration and module scaling

1. Use the design wrapper to obtain the module count, flow allocation and design parameters.
2. Check that the selected off-design bundle represents the intended nominal module design. The supplied HP bundle represents a 1 MW-class module. Applying it to another nominal capacity requires validation.
3. Supply per-module flow and heating load to the off-design wrappers. For identical active modules sharing the flow and load equally, divide the installation-level quantities by the number of active modules.
4. Pass the design expander stage count or design pressures to the ORC off-design wrapper.
5. Evaluate one representative module when all active modules have the same configuration and operating conditions. For different operating conditions, evaluate the modules separately.
6. Sum extensive quantities such as heat, power and mass flow to obtain installation totals. For identical operating modules, this is equivalent to multiplying their per-module values by the number of active modules. Temperatures, pressures, efficiencies, COP values and fractions retain their per-module values under identical operation.

The CLI examples illustrate independent wrapper calls. Use a consistent design and operating point when combining them in a system study.

The wrappers print JSON for integration with another Python program or external application. Prediction functions can also be imported from the supplied modules. Consult the corresponding source file for the function signatures.

## Validation and reproducibility

The examples follow the supplied wrapper documentation. Release checks should execute them with the corresponding Python files and model bundles and record their outputs.

The following release-specific documentation remains to be completed.

- Exact Python and dependency versions used to serialize and test the model bundles.
- Complete numerical input bounds for the design ANN and both ORC off-design bundles.
- Model accuracy metrics, data-splitting details and reference prediction outputs for these specific bundles.

The HP operating ranges, ORC topology-selection rule and HP energy-balance relationship are documented above. Use the model-specific warnings and reference-model comparisons when assessing a new operating point.

## Author and affiliation

**Tryfon Roumpedakis**

Laboratory of Applied Thermodynamics, School of Mechanical Engineering, National Technical University of Athens

ORCID [0000-0003-2107-1206](https://orcid.org/0000-0003-2107-1206)

## Licence

The source code and trained surrogate models are released under the Apache License 2.0. See `LICENSE` for the full terms. Redistributed copies must preserve the applicable licence and attribution notices.

Please acknowledge the author by citing the software version used, as described in the Citation section.
