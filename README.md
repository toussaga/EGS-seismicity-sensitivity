# EGS-seismicity-sensitivity

Code accompanying:

> Toussaint, G., Valley, B., Heinze, T. and Miller, S.A. (2026), "Stress regime and fracture criticality control maximum magnitude of induced seismicity in Enhanced Geothermal Systems: insights from a coupled thermo-hydro-mechanical model", submitted in International Journal of Rock Mechanics and Mining Sciences. 

This repository contains a 2D coupled thermo-hydro-mechanical (THM) model of injection-induced seismicity in fractured reservoirs (Enhanced Geothermal Systems), written in Julia, together with the scripts used to generate the fracture networks, run the simulations of the paper.

---

## Model overview

- **Hydraulics:** Darcy flow with fluid and pore compressibility.
- **Heat transport:** separate fluid and rock temperatures; conduction, advection, fluid–rock heat exchange and frictional heating.
- **Mechanics:** elasto-plastic dynamic-relaxation solver. Drucker-Prager plasticity in the rock matrix; Coulomb friction and tensile failure on fracture cells, resolved on each fracture's orientation.
- **Coupling:** stress- and failure-dependent permeability with healing, permeability-dependent porosity, poro- and thermo-elastic stresses.
- **Seismicity:** events are detected from jumps in the deviatoric strain rate and grouped into connected patches; the moment magnitude is computed from the net slip and the rupture size of each event.

The solver uses pseudo-transient iterations on a staggered grid using [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) .

---

## Repository structure

```
src/
  model_sensitivity.jl                    main script: sensitivity simulations
  domain_properties.jl                    material map, properties, injection cells
  fracture_load.jl                        reading fracture networks
  plastic_corrector_matrix.jl             plasticity, rock matrix
  plastic_corrector_faults.jl             plasticity, fracture cells
  stimulation_scenarios.jl                injection protocols, injected flow rate
  earthquake_detection.jl                 event detection and magnitudes
fractures/
  gen_frac_networks.jl                    generates the fracture networks
  fractures_*.csv                         networks used in the paper
benchmarks/
  benchmark_hydraulic.jl                  hydraulic benchmark
  benchmark_lauwerier.jl                  thermal benchmark
  model_basel_2d.jl                       main model applied to Basel EGS project             

---

## Requirements

- Julia (tested with 1.12.1)
- Julia packages (versions pinned in `Project.toml` / `Manifest.toml`): ParallelStencil, CUDA, GPUArrays, CSV, DataFrames, Distributions, LazyArrays, StaticArrays

---

## Running the simulations

### Sensitivity analysis

All scenario parameters are in the `SCENARIO SETTINGS` block at the top of
`model/model_sensitivity.jl`:

| Parameter | Meaning |
|---|---|
| `K_RATIO` | stress ratio k = σxx/σyy (NF: k < 1, SS: k ≈ 1, TF: k > 1) |
| `FRACTURE_FILE`, `FAMILY_PROPS` | fracture network and fracture hydraulic properties |
| `FRICTION_ANGLE`, `FAULT_FRICTION_FACTOR` | matrix friction angle; fracture friction angle = factor × matrix angle |
| `INJ_X`, `INJ_Y1`, `INJ_Y2` | injection interval (position and depths) |
| `OUT_DIR` | output folder |

### Fracture networks

```bash
cd networks
julia --project=.. gen_sensitivity_networks.jl
```

The script writes one network per stress regime (NF, SS, TF). The networks used in
the paper are included in `networks/`, because the random number stream can change
between Julia versions. The parameters of each network set are:

| Set | `LMIN`–`LMAX` (before rescaling) | `L_TARGET` | `ALPHA_PL` | `DMIN` | `SEED` |
|---|---|---|---|---|---|
| short06 | <…> | <…> | <…> | <…> | <…> |
| large06 | <…> | <…> | <…> | <…> | <…> |

Note: lengths are rescaled after sampling to reach the total length `L_TARGET`.
The script prints the actual length range of each network.

### Basel benchmark

`model/model_basel_2d.jl` uses the same model with the Basel domain, stress profile,
fracture network and measured injection pressure. **The injection pressure data
are not included** (see below). Once you have them, place the file in `data/`
and run:

```bash
cd model
julia --project=.. model_basel_2d.jl
```

Without the data file, the script stops with an explanatory message.

### Verification benchmarks

<Short description and how to run the THERMAID example 6 and Lauwerier benchmarks.>

---

## Outputs

Each run writes to `OUT_DIR`:

| File | Content |
|---|---|
| `Mw_catalog.csv` | event catalogue: time [s], Mw, x, y [m], rupture length [m] |
| `inj_data.csv` | time series: injected flow rate, well pressure, energy budget, number of failing cells |
| `*_itXXXXXX.csv`, `*_end.csv` | field snapshots (pressure, permeability, temperature, stresses, plastic strain) |

---

## Data availability

- **Basel injection pressure:** not redistributed here. Source: <reference>.
- **Basel stress profile:** <reference>.
- **Basel seismic catalogue:** <reference>.

---

## Citation

If you use this code, please cite the paper above and the archived code:

> <Authors> (<year>). <repo-name> (version <x.y>). Zenodo. https://doi.org/<zenodo-doi>

## License

This project is licensed under the MIT License. See the LICENSE file for details.

