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
  model_basel_2d.jl                       main model calibrated to Basel EGS project
  p_input_basel.jl                        function that reads the Basel injection pressure
  fractures/
    gen_frac_networks.jl                    generates the fracture networks
    fractures_*.csv                         networks used in the paper
benchmarks/
  benchmark_hydraulic.jl                  hydraulic benchmark
  benchmark_lauwerier.jl                  thermal benchmark
Manifest.toml
Project.toml
```
---

## Requirements

- Julia (tested with 1.12.1)
- Julia packages (versions pinned in `Project.toml` / `Manifest.toml`): ParallelStencil, CUDA, GPUArrays, CSV, DataFrames, Distributions, LazyArrays, StaticArrays

---
### Verification benchmarks

Requires the following packages:
- `benchmark_hydraulic.jl`: Printf, LinearAlgebra, Plots, Statistics, ParallelStencil
- `benchmark_lauwerier.jl`: Printf, LinearAlgebra, Plots, Statistics, SpecialFunctions, ParallelStencil

### Basel benchmark

`src/model_basel_2d.jl` uses the same model with the Basel domain, stress profile, fracture network and measured injection pressure. **The injection pressure data
are not included** (see below). Once you have them, place the file in the folder and run the code.

---

## Data availability

- **Basel injection pressure:** not redistributed here. Source: Häring, M.O., Schanz, U., Ladner, F., Dyer, B.C., 2008. Characterisation of the Basel 1 enhanced geothermal system. Geothermics 37, 469–495. https://doi.org/10.1016/j.geothermics.2008.06.002.
- **Basel stress profile:** Valley, B., Evans, K.F., 2019. Stress magnitudes in the Basel enhanced geothermal system. International Journal of Rock Mechanics and Mining
Sciences 118, 1–20. https://doi.org/10.1016/j.ijrmms.2019.03.008.
- **Basel seismic catalogue:** Herrmann, M., Kraft, T., Tormann, T., Scarabello, L., Wiemer, S., 2019. A Consistent High-Resolution Catalog of Induced Seismicity in
Basel Based on Matched Filter Detection and Tailored Post-Processing. Journal of Geophysical Research: Solid Earth 124, 8449–8477. https://doi.org/10.1029/2019JB017468

---

## Citation

If you use this code, please cite the paper above and the archived code:

> Toussaint, G., Valley, B., Heinze, T. and Miller, S.A. (2026). EGS-seismicity-sensitivity (version 1.0). Zenodo. [![DOI](https://zenodo.org/badge/1439607519.svg)](https://doi.org/10.5281/zenodo.23811851)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

