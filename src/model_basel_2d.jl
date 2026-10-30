# =============================================================================
#  model_basel_2d.jl  —  Basel benchmark (current 2D model)
#
#  Simulation of the 2006 Basel (Switzerland) EGS stimulation with the same 2D
#  THM model as the sensitivity study (model_sensitivity.jl). Only the inputs
#  differ: domain, Basel stress profile, fracture network, and the measured
#  injection pressure.
#
#  Accompanies: Toussaint, G., Valley, B., Heinze, T. and Miller, S.A. (2026), "Stress regime and fracture criticality 
#  control maximum magnitude of induced seismicity in Enhanced Geothermal Systems: insights from a coupled 
#  thermo-hydro-mechanical model", submitted in International Journal of Rock Mechanics and Mining Sciences. 
#
#  ---------------------------------------------------------------------------
#  !! INJECTION PRESSURE DATA ARE NOT INCLUDED !!
#  ---------------------------------------------------------------------------
#  The model is driven by the measured injection pressure of the Basel
#  stimulation (function simulate_real_pressure!, file p_input_basel.jl).
#  These data are not ours and cannot be redistributed with this code.
#    Source : <reference of the data / data provider>
#    Access : <how to request or download them>
#  To run this benchmark, obtain the data, convert them to the format
#  described in p_input_basel.jl, and set BASEL_PRESSURE_FILE below.
#  Without this file the script stops with an explanatory error.
#  ---------------------------------------------------------------------------
#
#  Differences with model_sensitivity.jl (all in the SETTINGS block / setup):
#    - domain 2.0 × 2.8 km, y = -5800 m to -3000 m (3.0–5.8 km depth)
#    - initial stress: Basel principal stresses (Sv, SH_max, Sh_min) rotated
#      into the vertical section of azimuth CS_ORIENTATION. With the default
#      angles the section is parallel to Sh_min, so σxx = Sh_min, σyy = Sv and
#      τxy = 0; other orientations give a non-zero initial shear stress τxy.
#    - injection: measured Basel pressure, applied at every time step
#    The physics, solver, plastic correctors, permeability model and event
#    detection are identical to the sensitivity simulations.
#
#  Required files (same folder)
#    domain_properties.jl,
#    plastic_corrector_matrix_newit_corr.jl, plastic_corrector_faults_2D_it_corr.jl,
#    stimulation_scenarios.jl, earthquake_detection_algo_new3.jl,
#    fracture_generator_2d.jl, p_input_basel.jl
#    + fracture network CSV in the 2D end-point format (FRACTURE_FILE)
#    + Basel injection pressure data (BASEL_PRESSURE_FILE, not included)
#
#  Conventions
#    - SI units (m, s, Pa, K) unless stated otherwise.
#    - x horizontal, y vertical and POSITIVE UPWARD.
#    - Stresses: compression is NEGATIVE.
#    - Fracture angle from vertical, positive = dips towards +x;
#      normal n = (cos α, sin α), down-dip tangent t = (sin α, -cos α).
#
#  Outputs (in OUT_DIR): same files as model_sensitivity.jl.
# =============================================================================

const USE_GPU = false     # NOTE: the script uses CuArray/CUDA calls directly,
                         # so it requires a CUDA GPU.
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using CUDA, GPUArrays

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
end

using Printf, LazyArrays, LinearAlgebra, Distributions, Random, CSV, DataFrames, StaticArrays
include("domain_properties.jl")
include("plastic_corrector_matrix.jl")
include("plastic_corrector_faults.jl")
include("stimulation_scenarios.jl")
include("earthquake_detection.jl")
include("fracture_load.jl")
include("p_input_basel.jl")

# =============================================================================
#  SETTINGS
# =============================================================================

# --- Output -------------------------------------------------------------------
const OUT_DIR = "basel2D_large6"   # folder where all results are written

# --- Injection pressure data (NOT INCLUDED, see header) ------------------------
const BASEL_PRESSURE_FILE = "WHP_basel.csv"   # must match the path read in p_input_basel.jl

# --- Basel stress profile ------------------------------------------------------
# Principal stresses as functions of y [m] (y < 0, compression negative):
#   Sv     = SV_GRADIENT · y
#   SH_max = SHMAX_GRADIENT · y + SHMAX_OFFSET
#   Sh_min = SHMIN_GRADIENT · y + SHMIN_OFFSET
# Source of the stress profile: <reference>
const SV_GRADIENT    = 24.9e3              # [Pa/m]
const SHMAX_GRADIENT, SHMAX_OFFSET = 5e3, -90e6   # [Pa/m], [Pa]
const SHMIN_GRADIENT, SHMIN_OFFSET = 7e3, -42e6   # [Pa/m], [Pa]

# Orientation of the principal stresses and of the model section
const CS_ORIENTATION = 54.0                # azimuth of the vertical model section [°]
const EULER_ANGLES   = (54.0, 90.0, 0.0)   # (a, b, c) Euler angles of the principal stress frame [°]

# --- Fracture network ------------------------------------------------------------
# CSV in the 2D end-point format read by load_fractures_2d
# (columns x1, y1, x2, y2, dip, strike, family; see fracture_generator_2d.jl).
const FRACTURE_FILE      = "fracture_new_large6.csv"
const FRACTURE_THICKNESS = 12.0 # [m] max. distance between a cell centre and a fracture
                                #     trace for the cell to belong to that fracture

# Hydraulic properties per fracture family: [k_0 (m²), 1/σ* (1/Pa), healing rate α]
const FAMILY_PROPS = Dict(
    1 => [1e-13, 1/70e6, 0.02],
    2 => [1e-13, 1/70e6, 0.02],
    3 => [1e-13, 1/70e6, 0.02],
)
const MATRIX_PROPS = [1e-15, 1/70e6, 0.02]   # same entries for the rock matrix

# --- Friction --------------------------------------------------------------------
const FRICTION_ANGLE        = 44.0   # matrix friction angle [°]
const FAULT_FRICTION_FACTOR = 0.7    # fracture friction angle = 0.7 × matrix angle

# --- Injection interval ------------------------------------------------------------
# One column of cells at x = INJ_X, between depths INJ_Y1 and INJ_Y2 (positive) [m]
const INJ_X  = 2000.0/2 - 5
const INJ_Y1 = 4354.0
const INJ_Y2 = 4754.0

# Cell where the yield function and strain rate are recorded (columns "F", "strain").
# The well pressure ("Pwell") is recorded at the centre of the injection interval.
const MONITOR_CELL = (102, 82)

# --- Timing ----------------------------------------------------------------------------
const T_TOTAL      = 24*60*60*10.56    # total simulated time [s]
const DETECT_AFTER = 40.0              # event detection starts after this time [s]

# --- Reproducibility -----------------------------------------------------------------
const RNG_SEED = 1234                  # background matrix orientations; `nothing` = unseeded

# --- Outputs -----------------------------------------------------------------------------
const SAVE_BURST   = false                 # high-frequency snapshots (Coulomb stress analysis)
const BURST_WINDOW = (240_000, 360_000)    # iteration range
const BURST_EVERY  = 2000
const SNAP_EVERY   = 120_000               # regular snapshots every N iterations

if !isfile(BASEL_PRESSURE_FILE)
    error("""
    Basel injection pressure data not found at '$BASEL_PRESSURE_FILE'.
    These data cannot be redistributed with this code. See the header of
    model_basel_2d.jl (and README) for the source and how to obtain them, and
    p_input_basel.jl for the expected format.""")
end

# =============================================================================
#  Helpers
# =============================================================================
@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

# Rotates the principal stress tensor diag(S1, S2, S3) by the Euler angles
# (a, b, c) [°], then into the vertical section of azimuth CS_orientation [°],
# and returns the in-plane components (σxx, σyy, τxy) of the section
# (x horizontal along the section, y vertical).
function project_stress_on_section(S1, S2, S3, a, b, c, CS_orientation)
    R = [cosd(a) * cosd(b)                                sind(a) * cosd(b)                                -sind(b)
         cosd(a) * sind(b) * sind(c) - sind(a) * cosd(c)  sind(a) * sind(b) * sind(c) + cosd(a) * cosd(c)  cosd(b) * sind(c)
         cosd(a) * sind(b) * cosd(c) + sind(a) * sind(c)  sind(a) * sind(b) * cosd(c) - cosd(a) * sind(c)  cosd(b) * cosd(c)]
    stress_xyz = R' * Diagonal([S1, S2, S3]) * R
    R_CS = [cosd(CS_orientation) -sind(CS_orientation) 0;
            sind(CS_orientation)  cosd(CS_orientation) 0;
            0                     0                    1]
    stress_CS = R_CS' * stress_xyz * R_CS
    return stress_CS[1, 1], stress_CS[3, 3], stress_CS[1, 3]
end

# =============================================================================
#  Kernels
# =============================================================================

# Normal and shear stress resolved on the local fracture/cell orientation.
# angle_2d [°] is measured from the vertical, positive = dips towards +x.
# Normal n = (cos α, sin α), down-dip tangent t = (sin α, -cos α)
# (same convention as the fracture corrector and the event detection).
@parallel_indices (ix,iy) function stress_on_faults_2d!(Sxx_dev, Syy_dev, Txy_dev, P, σn, τ, angle_2d)
    # full 2D stress tensor (deviatoric + mean stress)
    σxx = Sxx_dev[ix,iy] + P[ix,iy]
    σyy = Syy_dev[ix,iy] + P[ix,iy]
    σxy = Txy_dev[ix,iy]

    # fault normal
    α  = angle_2d[ix,iy] * π / 180.0
    nx = cos(α)
    ny = sin(α)

    # traction t = σ·n
    tx = σxx * nx + σxy * ny
    ty = σxy * nx + σyy * ny

    # normal stress σn = t·n, shear stress τ = |t - σn n|
    σn_val = tx * nx + ty * ny
    τx = tx - σn_val * nx
    τy = ty - σn_val * ny

    σn[ix,iy] = σn_val
    τ[ix,iy]  = sqrt(τx*τx + τy*τy)
    return
end

@parallel function assign!(A, B)
    @all(A) = @all(B)
    return
end

# Permeability evolution:
#  - failing cell  : permeability jumps towards 200 × initial value, healing clock reset
#  - otherwise     : exponential healing back to k_ini with rate α_heal
#  - always        : stress dependence k = k0 * exp(-|σn| / σ*)
@parallel_indices (ix,iy) function update_permeability!(F_fail, trig_tog, σn, theal, kc, kc0, k_ini, α_heal, _σ_star, dt)
    if F_fail[ix,iy] > 0
        theal[ix,iy]    = 0
        trig_tog[ix,iy] = 1
        kc0[ix,iy] += (k_ini[ix,iy] * 200 - kc0[ix,iy]) * 0.5
    else
        trig_tog[ix,iy] = 0
        theal[ix,iy] = theal[ix,iy] + dt
        kc0[ix,iy]   = k_ini[ix,iy] + (kc0[ix,iy] - k_ini[ix,iy]) * exp(-α_heal[ix,iy] * theal[ix,iy]/86400 * (dt/86400))
    end
    kc[ix,iy] = kc0[ix,iy] * exp(-min(abs(σn[ix,iy]), 1e8) * _σ_star[ix,iy] * (dt/86400))
    return
end

# Porosity from permeability (ϕ ∝ k^0.2) and update of all derived coefficients
@parallel function update_porosity!(ϕ, ϕ0, kc, k_ini, k_η, k_η_βϕ, η, _βϕ, _β, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, λ_ρCp_f, λ_ρCp_r, _ϕρCp_fluid, ρCp_fluid, _1ϕρCp_rock, ρCp_rock)
    @all(ϕ)           = @all(ϕ0) * (@all(kc) / @all(k_ini)) ^ 0.2
    @all(_βϕ)         = _β / @all(ϕ)
    @all(k_η)         = @all(kc) / @all(η)
    @all(k_η_βϕ)      = @all(k_η) * @all(_βϕ)
    @all(λ_ρCp_ϕ_f)   = λ_ρCp_f / @all(ϕ)
    @all(λ_ρCp_1ϕ_r)  = λ_ρCp_r / (1 - @all(ϕ))
    @all(_ϕρCp_fluid) = 1 / ρCp_fluid / @all(ϕ)
    @all(_1ϕρCp_rock) = 1 / ρCp_rock / (1 - @all(ϕ))
    return
end

# ---------------------------------------------------------------- Hydraulics ---
@parallel function compute_iter_params_P!(dτ_ρ, k_η_βϕ, _Re, Vpdτ, max_lxy)
    @all(dτ_ρ)  = Vpdτ * max_lxy * _Re / @maxloc(k_η_βϕ)
    return
end

# Darcy flux: qD (pseudo-transient, damped) and qD2 (physical flux)
@parallel function compute_Darcyflux!(qDx, qDy, qDx2, qDy2, Pf, Tf, k_η, θr_dτ, _1_θr_dτ , _dx, _dy, αρgy)
    @all(qDx)  = (@all(qDx) * θr_dτ - @av_xi(k_η) * (@d_xi(Pf) * _dx                    )) * _1_θr_dτ
    @all(qDy)  = (@all(qDy) * θr_dτ - @av_yi(k_η) * (@d_yi(Pf) * _dy - αρgy * @av_yi(Tf))) * _1_θr_dτ
    @all(qDx2) = -@av_xi(k_η) * (@d_xi(Pf) * _dx                   )
    @all(qDy2) = -@av_yi(k_η) * (@d_yi(Pf) * _dy - αρgy * @av_yi(Tf))
    return
end

@parallel function update_Pf!(Pf, Pold, qDx, qDy, dτ_ρ, _dt, _dx, _dy, _βϕ)
    @inn(Pf) = (@inn(Pf) +  @all(dτ_ρ) * (@inn(Pold) * _dt - (@d_xa(qDx) * _dx * @inn(_βϕ) + @d_ya(qDy) * _dy * @inn(_βϕ)))) / (1.0 + @all(dτ_ρ) * _dt)
    return
end

# Interstitial fluid velocity magnitude
@parallel function flux_velocity(v, qDx2, qDy2, ϕ)
    @all(v) = sqrt((@av_xa(qDx2) / @inn(ϕ))^2 + (@av_ya(qDy2) / @inn(ϕ))^2)
    return
end

# ------------------------------------------------------------------- Thermics ---
@parallel function compute_iter_params_T!(dτ_ρTf, dτ_ρTr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, _ReTf, _ReTr, Vpdτ, max_lxy)
    @all(dτ_ρTf)  = Vpdτ * max_lxy * _ReTf / @maxloc(λ_ρCp_ϕ_f)
    @all(dτ_ρTr)  = Vpdτ * max_lxy * _ReTr / @maxloc(λ_ρCp_1ϕ_r)
    return
end

# Frictional heating rate: βT / (ρCp) · ε̇2_pl · τ2
@parallel function compute_Tfrictional!(T_frictional, βT_ρCp, ϵ, τ_heat)
    @inn(T_frictional) = βT_ρCp * @inn(ϵ) * @all(τ_heat)
    return
end

# Conductive heat fluxes in fluid (Tf) and rock (Tr)
@parallel function compute_Tflux!(qTfx, qTfy, qTrx, qTry, qTfx2, qTfy2, qTrx2, qTry2, Tf, Tr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, θTf_dτ, θTr_dτ, _1_θTf_dτ, _1_θTr_dτ, _dx, _dy)
    @all(qTfx)   = (@all(qTfx) * θTf_dτ - @av_xi(λ_ρCp_ϕ_f) * @d_xi(Tf) * _dx) * _1_θTf_dτ
    @all(qTfy)   = (@all(qTfy) * θTf_dτ - @av_yi(λ_ρCp_ϕ_f) * @d_yi(Tf) * _dy) * _1_θTf_dτ
    @all(qTfx2)  = - @av_xi(λ_ρCp_ϕ_f) * @d_xi(Tf) * _dx
    @all(qTfy2)  = - @av_yi(λ_ρCp_ϕ_f) * @d_yi(Tf) * _dy
    @all(qTrx)   = (@all(qTrx) * θTr_dτ - @av_xi(λ_ρCp_1ϕ_r) * @d_xi(Tr) * _dx) * _1_θTr_dτ
    @all(qTry)   = (@all(qTry) * θTr_dτ - @av_yi(λ_ρCp_1ϕ_r) * @d_yi(Tr) * _dy) * _1_θTr_dτ
    @all(qTrx2)  = - @av_xi(λ_ρCp_1ϕ_r) * @d_xi(Tr) * _dx
    @all(qTry2)  = - @av_yi(λ_ρCp_1ϕ_r) * @d_yi(Tr) * _dy
    return
end

# Upwind advection of fluid temperature
@parallel_indices (ix,iy) function advect_T!(dTdt, T, qDx2, qDy2, _dx, _dy, ϕ)
    if (ix<=size(dTdt, 1) && iy<=size(dTdt, 2))
        dTdt[ix, iy] = ((qDx2[ix, iy]   > 0)  * qDx2[ix, iy]   * (T[ix+1, iy+1] - T[ix, iy+1])   * _dx +
                        (qDx2[ix+1, iy] < 0)  * qDx2[ix+1, iy] * (T[ix+2, iy+1] - T[ix+1, iy+1]) * _dx +
                        (qDy2[ix, iy]   > 0)  * qDy2[ix, iy]   * (T[ix+1, iy+1] - T[ix+1, iy])   * _dy +
                        (qDy2[ix, iy+1] < 0)  * qDy2[ix, iy+1] * (T[ix+1, iy+2] - T[ix+1, iy+1]) * _dy) / @inn(ϕ)
    end
    return
end

# Fluid-rock heat exchange: empirical heat-transfer coefficient h(aperture, velocity)
@parallel function T_transfer!(Q, h, a, dx, Tf, Tr, v)
    @all(h) = 0.00905 * ((0.0009217 + exp(-2.314e5 * @inn(a)))^(-1) * (0.0693 + exp(-136.9 * @all(v)))^(-1)) + 650 * @all(v)
    @all(Q) = @all(h) * 2 * dx * (@inn(Tf) - @inn(Tr))
    return
end

@parallel function update_T!(Tf, Tr, Tfold, Trold, qTfx, qTfy, qTrx, qTry, dTdt, _dt, dτ_ρTf, dτ_ρTr, T_frictional, Q, ϕ, _ϕρCp_fluid, _1ϕρCp_rock, _dx, _dy)
    @inn(Tf) = (@inn(Tf) +  @all(dτ_ρTf) * (@inn(Tfold) * _dt - (@d_xa(qTfx) * _dx + @d_ya(qTfy) * _dy) - @all(Q) * @inn(_ϕρCp_fluid) - @all(dTdt)                        )) / (1.0 + @all(dτ_ρTf) * _dt)
    @inn(Tr) = (@inn(Tr) +  @all(dτ_ρTr) * (@inn(Trold) * _dt - (@d_xa(qTrx) * _dx + @d_ya(qTry) * _dy) + @all(Q) * @inn(_1ϕρCp_rock) + @inn(T_frictional) / (1 - @inn(ϕ)))) / (1.0 + @all(dτ_ρTr) * _dt)
    return
end

# Residuals of the P, Tf and Tr equations (convergence check)
@parallel function check_res!(ResP, ResTf, ResTr, Pf, Pold, qDx2, qDy2, Tf, Tr, Tfold, Trold, dTdt, T_frictional, qTfx2, qTfy2, qTrx2, qTry2, _dt, _dx, _dy, _βϕ, Q, _ϕρCp_fluid, _1ϕρCp_rock, ϕ)
    @all(ResP)  = -(@inn(Pf) - @inn(Pold))  * _dt - (@d_xa(qDx2) * _dx  * @inn(_βϕ) + @d_ya(qDy2) * _dy * @inn(_βϕ))
    @all(ResTf) = -(@inn(Tf) - @inn(Tfold)) * _dt - (@d_xa(qTfx2) * _dx             + @d_ya(qTfy2) * _dy           ) - @all(Q) * @inn(_ϕρCp_fluid) - @all(dTdt)
    @all(ResTr) = -(@inn(Tr) - @inn(Trold)) * _dt - (@d_xa(qTrx2) * _dx             + @d_ya(qTry2) * _dy           ) + @all(Q) * @inn(_1ϕρCp_rock) + @inn(T_frictional) / (1 - @inn(ϕ))
    return
end

# ----------------------------------------------------------------- Mechanics ---
# Builds stresses (incl. pore pressure) from the total-stress arrays and splits
# them into mean stress P0 and deviatoric parts. Used at start-up and every step.
@parallel function initial_stress!(σxx, σyy, τxy, σxx_tot, σyy_tot, τxy_tot, σxx_dev, σyy_dev, τxy_dev, P_hydro, P0, Pf)
    @all(σxx) = @inn(σxx_tot) + @all(P_hydro) + @all(Pf)
    @all(σyy) = @inn(σyy_tot) + @all(P_hydro) + @all(Pf)
    @all(τxy) = @inn(τxy_tot)
    @all(P0)  = 0.5 * (@all(σxx) + @all(σyy))
    @all(σxx_dev) = @all(σxx) - @all(P0)
    @all(σyy_dev) = @all(σyy) - @all(P0)
    @all(τxy_dev) = @all(τxy)
    return
end

# Total stress = stress - pore pressure (hydrostatic + overpressure) - thermal stress
#   K: bulk modulus [Pa], α_rock: thermal expansion of rock [1/K]
@parallel function total_stress!(σxx, σyy, τxy, σxx_tot, σyy_tot, τxy_tot, Pf, P_hydro, T, Told, K, α_rock)
    @inn(σxx_tot) = @all(σxx) - @all(Pf) - @all(P_hydro) - K * α_rock * (@all(T) - @all(Told))
    @inn(σyy_tot) = @all(σyy) - @all(Pf) - @all(P_hydro) - K * α_rock * (@all(T) - @all(Told))
    @inn(τxy_tot) = @all(τxy)
    return
end

# Damped momentum balance (dynamic relaxation, Rozhko), including rock weight
@parallel function velocity_update(σxx_tot, σyy_tot, τxy_tot, Vx, Vy, _dx, _dy, dt, damp, ρ_rock, _ρ_elastic,σxx_dx, σyy_dy, τxy_dx, τxy_dy )
    @all(σxx_dx) = @d_xa(σxx_tot) * _dx
    @all(σyy_dy) = @d_ya(σyy_tot) * _dy
    @all(τxy_dx) = @d_xa(τxy_tot) * _dx
    @all(τxy_dy) = @d_ya(τxy_tot) * _dy
    @all(Vx) = @all(Vx) * (1 - damp) + dt * (@av_ya(σxx_dx) + @av_xa(τxy_dy)) * _ρ_elastic
    @all(Vy) = @all(Vy) * (1 - damp) + dt * (@av_ya(τxy_dx) + @av_xa(σyy_dy) - (9.81 * ρ_rock)) * _ρ_elastic
    return
end

@parallel function update_displacement!(Ux, Uy, Vx, Vy, dt)
    @all(Ux) = @all(Ux) + dt * @all(Vx)
    @all(Uy) = @all(Uy) + dt * @all(Vy)
    return
end

function compute_strainrate_tensor!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
    εxx .= (diff(Vx[:, 1:end-1], dims = 1) .+ diff(Vx[:, 2:end], dims = 1)) .* _dx ./ 2
    εyy .= (diff(Vy[1:end-1, :], dims = 2) .+ diff(Vy[2:end, :], dims = 2)) .* _dy ./ 2
    εxy .= 0.5.*((diff(Vx[1:end-1, :], dims = 2) .+ diff(Vx[2:end, :], dims = 2)) .* _dy ./ 2 + (diff(Vy[:, 1:end-1], dims = 1) .+ diff(Vy[:, 2:end], dims = 1)) .* _dx ./ 2)
    return εxx, εyy, εxy
end

# Deviatoric strain rate and its second invariant ε̇2_dev
@parallel function compute_deviatoric_strainrate!(εxx_dev, εyy_dev, εxy_dev, εxx, εyy, εxy, divergence,ε2_dev)
    @all(divergence) = (@all(εxx) + @all(εyy))
    @all(εxx_dev) = @all(εxx) - 1/3 * @all(divergence)
    @all(εyy_dev) = @all(εyy) - 1/3 * @all(divergence)
    @all(εxy_dev) = @all(εxy)
    @all(ε2_dev) = sqrt(0.5*(@all(εxx_dev)^2 + @all(εyy_dev)^2) + @all(εxy_dev)^2)
    return
end

# Elastic trial stress (G: shear modulus, K: bulk modulus)
@parallel function compute_stress_predictor!(σxx_check, σyy_check, τxy_check, σxx_dev, σyy_dev, τxy_dev,P, P0, dt, G, K, εxx_dev, εyy_dev, εxy_dev, divergence)
    @all(σxx_check) = @all(σxx_dev) + dt * 2 * G * @all(εxx_dev)
    @all(σyy_check) = @all(σyy_dev) + dt * 2 * G * @all(εyy_dev)
    @all(τxy_check) = @all(τxy_dev) + dt * 2 * G * @all(εxy_dev)
    @all(P) = @all(P0) + K * dt * @all(divergence)
    return
end

@parallel function compute_effective_stress!(Sxx_dev, Syy_dev, Txy_dev, Sxx_eff, Syy_eff, Txy_eff, P)
    @all(Sxx_eff) = @all(Sxx_dev) + @all(P)
    @all(Syy_eff) = @all(Syy_dev) + @all(P)
    @all(Txy_eff) = @all(Txy_dev)
    return
end

# Zero-gradient boundary conditions
@parallel_indices (iy) function bc_x!(A)
    A[1  , iy] = A[2    , iy]
    A[end, iy] = A[end-1, iy]
    return
end

@parallel_indices (ix) function bc_y!(A)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

# =============================================================================
#  Physical parameters
# =============================================================================
lx, ly     = 2.0e3, 2.8e3            # Domain size                         [m]
α_fluid    = 7.5e-4                  # Thermal expansion of water          [1/K]
α_rock     = 1.0e-5                  # Thermal expansion of rock           [1/K]
ρ_fluid    = 1000                    # Density of water                    [kg/m3]
ρ_rock     = 2700                    # Density of rock                     [kg/m3]
gy         = 9.81                    # Gravitational acceleration          [m/s2]
ϕ_rock     = 0.03                    # Rock porosity                       [-]
k_rock     = MATRIX_PROPS[1]         # Rock permeability                   [m2]
β_fluid    = 1e-10                   # Fluid compressibility               [1/Pa]
β_phi      = 1e-8                    # Pore compressibility                [1/Pa]
ΔT         = 35                      # Geothermal gradient                 [K/km]
λ_rock     = 2                       # Thermal conductivity (limestone)    [W/m/K]
λ_fluid    = 0.6                     # Thermal conductivity (water)        [W/m/K]
Cp_fluid   = 4200                    # Specific heat capacity (water)      [J/kg/K]
Cp_rock    = 910                     # Specific heat capacity (limestone)  [J/kg/K]
βT         = 1                       # Taylor-Quinney coefficient          [-]

# Mechanics
poisson        = 0.22                    # Poisson ratio                   [-]
E              = 65.0e9                  # Young modulus                   [Pa]
G              = E / 2 / (1 + poisson)   # Shear modulus                   [Pa]
K              = E / 3 / (1-2*poisson)   # Bulk modulus                    [Pa]
friction_angle = FRICTION_ANGLE          # Matrix friction angle           [°]
coh, coh_f     = 25e6, 100               # Cohesion matrix / fractures     [Pa]
tensile        = 4e6                     # Tensile strength                [Pa]
ψ              = 10.0                    # Dilation angle (matrix)         [°]
phi_f          = friction_angle * FAULT_FRICTION_FACTOR   # Fracture friction angle [°]
psi_f          = ψ / 1.75                # Fracture dilation angle         [°]
seismic_tr     = 1e-11                   # Threshold on Δε̇2_dev for seismic detection
T_INJ_FLUID    = 60.0 + 273.15           # Injection temperature           [K]

# Derived
αρgy       = α_fluid * ρ_fluid * gy
ρCp_fluid  = ρ_fluid * Cp_fluid
ρCp_rock   = ρ_rock  * Cp_rock
λ_ρCp_f    = λ_fluid / ρCp_fluid
λ_ρCp_r    = λ_rock / ρCp_rock
_β         = 1 / (β_fluid + β_phi)
ρCp_ratio  = ρCp_fluid / ρCp_rock
βT_ρCp     = βT / ρCp_rock

# =============================================================================
#  Numerical parameters
# =============================================================================
nx       = 200
ny       = ceil(Int, nx * ly / lx)
ttot     = T_TOTAL
dt_diff  = 1

tol        = 2e-8                # PT iteration tolerance
itMax      = 3000                # max PT iterations per time step
nout       = 5                   # residual check every nout iterations
CFL        = 1/sqrt(2)
adv_lim    = ϕ_rock/ρCp_ratio    # advection CFL limit
_ρ_elastic = 1 / 1.0e9           # inverse computational density (Rozhko)
damp       = 1e-1                # elastic damping (Rozhko); reduced to 0.08 once failure starts
Vp         = sqrt(maximum(9 * K * G / (3 * K + G)) * _ρ_elastic)  # wave velocity
time       = 0

dx, dy   = lx / nx, ly / ny
_dx, _dy = 1/dx, 1/dy
Vpdτ     = CFL * min(dx, dy)
max_lxy  = max(lx, ly)
xn, yn   = LinRange(0, lx, nx + 1), LinRange(-5800, -3000, ny + 1)
xc, yc   = av1(xn), av1(yn)
dt_mech  = 1 / 4 * min(dx, dy) / Vp / 2
dt_eff   = min(dt_diff, dt_mech)

mkpath(OUT_DIR)
@printf("Monitoring cell (F, strain) at x = %.1f m, y = %.1f m\n", xc[MONITOR_CELL[1]], yc[MONITOR_CELL[2]])

# =============================================================================
#  Fracture network and material properties
# =============================================================================
RNG_SEED === nothing || Random.seed!(RNG_SEED)

# Orientation of each cell, angle from vertical [°]. Background matrix: random
# around 45°; fracture cells are overwritten by assign_geometry_2d! below.
angle_2d            = rand(Normal(45, 5), nx, ny)
let cols = propertynames(CSV.read(FRACTURE_FILE, DataFrame; limit = 1))
    all(c -> c in cols, (:x1, :y1, :x2, :y2, :family)) || error("""
        $FRACTURE_FILE is not in the 2D end-point format (x1, y1, x2, y2, dip, strike, family).
        Convert it first with convert_basel_network.jl.""")
end
Faults, families    = load_fractures_2d(FRACTURE_FILE, FRACTURE_THICKNESS)
num_faults          = length(Faults)

inj_points          = get_injection_points(xc, yc, nx, ny, INJ_X, INJ_Y1, INJ_Y2, dx/2)
isempty(inj_points) && error("No injection cell found: check INJ_X, INJ_Y1, INJ_Y2")
MONITOR_WELL        = inj_points[cld(length(inj_points), 2)]   # centre of the injection interval
@printf("Monitoring cell (well) at x = %.1f m, y = %.1f m\n", xc[MONITOR_WELL[1]], yc[MONITOR_WELL[2]])
kc0, _σ_star, α_heal = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)

# properties table: row 1 = matrix, row i+1 = fracture i (from its family)
properties      = Array{Float64, 2}(undef, num_faults+1, 3)
properties[1,:] = MATRIX_PROPS
for (i,fam) in enumerate(families)
    properties[i+1,:] = FAMILY_PROPS[fam]
end
material = assign_material(Faults, xc, yc, nx, ny)
assign_properties!(material, properties, kc0, _σ_star, α_heal)
assign_geometry_2d!(material, Faults, angle_2d)
CSV.write(joinpath(OUT_DIR, "angle_2d.csv"), Tables.table(angle_2d))
CSV.write(joinpath(OUT_DIR, "material.csv"), Tables.table(material))   # 1 = matrix, >1 = fracture (used by the CFS scripts)

material_d = CuArray(material)
dip_d      = CuArray(angle_2d)

# =============================================================================
#  Initial stress state: Basel principal stresses projected onto the section
#    x horizontal along the section, y vertical (positive upward),
#    compression negative. τxy = 0 when the section is parallel to a
#    principal direction (the default here: σxx = Sh_min, σyy = Sv).
# =============================================================================
P_hydro_y  = .-(ρ_fluid .* gy .* yc)              # hydrostatic pressure [Pa]
P_hydro    = CuArray(repeat(P_hydro_y', nx, 1))

σxx_ini, σyy_ini, τxy_ini = zeros(nx, ny), zeros(nx, ny), zeros(nx, ny)
for j in 1:ny
    Sv    = SV_GRADIENT * yc[j]
    SHmax = SHMAX_GRADIENT * yc[j] + SHMAX_OFFSET
    Shmin = SHMIN_GRADIENT * yc[j] + SHMIN_OFFSET
    sxx, syy, sxy = project_stress_on_section(Sv, SHmax, Shmin, EULER_ANGLES..., CS_ORIENTATION)
    σxx_ini[:, j] .= sxx
    σyy_ini[:, j] .= syy
    τxy_ini[:, j] .= sxy
end

# =============================================================================
#  Array allocation
# =============================================================================
σn, τ                           = @zeros(nx, ny), @zeros(nx, ny)
σxx_tot, σyy_tot, τxy_tot       = @zeros(nx+2, ny+2), @zeros(nx+2, ny+2), @zeros(nx+2, ny+2)
σxx_cpu, σyy_cpu, τxy_cpu       = zeros(nx+2, ny+2), zeros(nx+2, ny+2), zeros(nx+2, ny+2)
σxx, σyy, τxy                   = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
σxx_check, σyy_check, τxy_check = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
σxx_dev, σyy_dev, τxy_dev       = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)

# hydraulics
qDx, qDy     = @zeros(nx - 1, ny - 2), @zeros(nx - 2, ny - 1)
qDx2, qDy2   = @zeros(nx - 1, ny - 2), @zeros(nx - 2, ny - 1)
ResP         = @zeros(nx - 2, ny - 2)
dτ_ρ         = @zeros(nx - 2, ny - 2)
v            = @zeros(nx - 2, ny - 2)
h            = @zeros(nx - 2, ny - 2)

# thermics
qTfx, qTfy     = @zeros(nx - 1, ny - 2), @zeros(nx - 2, ny - 1)
qTfx2, qTfy2   = @zeros(nx - 1, ny - 2), @zeros(nx - 2, ny - 1)
qTrx, qTry     = @zeros(nx - 1, ny - 2), @zeros(nx - 2, ny - 1)
qTrx2, qTry2   = @zeros(nx - 1, ny - 2), @zeros(nx - 2, ny - 1)
dTdt           = @zeros(nx - 2, ny - 2)
T_frictional   = @zeros(nx, ny)
ResTf, ResTr   = @zeros(nx - 2, ny - 2), @zeros(nx - 2, ny - 2)
Q              = @zeros(nx - 2, ny - 2)
dτ_ρTf, dτ_ρTr = @zeros(nx - 2, ny - 2), @zeros(nx - 2, ny - 2)

# mechanics
Pold, Pf                  = @zeros(nx, ny), @zeros(nx, ny)
P0, P_stress, divergence  = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
Vx, Vy                    = @zeros(nx + 1, ny + 1), @zeros(nx + 1, ny + 1)
Ux, Uy                    = @zeros(nx + 1, ny + 1), @zeros(nx + 1, ny + 1)
εxx, εyy, εxy             = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
εxx_dev, εyy_dev, εxy_dev = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
F_type, F_check, F_fail   = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
Tau2                      = @zeros(nx, ny)
ε2_old, ε2_dev, ε2_pl, ε2_el = @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny), @zeros(nx, ny)
σxx_dx, σyy_dy            = @zeros(nx + 1, ny + 2), @zeros(nx + 2, ny + 1)
τxy_dx, τxy_dy            = @zeros(nx + 1, ny + 2), @zeros(nx + 2, ny + 1)

# =============================================================================
#  Initialisation
# =============================================================================
σxx_cpu[2:end-1, 2:end-1] .= σxx_ini
σyy_cpu[2:end-1, 2:end-1] .= σyy_ini
τxy_cpu[2:end-1, 2:end-1] .= τxy_ini

# stresses imposed on the top boundary during the run
bc_y  = σyy_cpu[nx ÷ 2, end-1]
bc_x  = σxx_cpu[nx ÷ 2, end-1]
bc_xy = τxy_cpu[nx ÷ 2, end-1]

copyto!(σxx_tot, σxx_cpu); copyto!(σyy_tot, σyy_cpu); copyto!(τxy_tot, τxy_cpu)
@parallel (1:size(σxx_tot, 2)) bc_x!(σxx_tot); @parallel (1:size(σyy_tot, 2)) bc_x!(σyy_tot); @parallel (1:size(τxy_tot, 2)) bc_x!(τxy_tot)
@parallel (1:size(σxx_tot, 1)) bc_y!(σxx_tot); @parallel (1:size(σyy_tot, 1)) bc_y!(σyy_tot); @parallel (1:size(τxy_tot, 1)) bc_y!(τxy_tot)
@parallel initial_stress!(σxx, σyy, τxy, σxx_tot, σyy_tot, τxy_tot, σxx_dev, σyy_dev, τxy_dev, P_hydro, P0, Pf)
@parallel stress_on_faults_2d!(σxx_dev, σyy_dev, τxy_dev, P0, σn, τ, dip_d)

# temperature: 15 °C at surface + geothermal gradient
T_y  = @. 15 + 273.15 + ΔT / 1000 * -yc
Tr   = CuArray(repeat(T_y, 1, nx)')
Tf   = copy(Tr)
# fluid viscosity decreasing linearly with depth
η_y  = 1e-3 .- 0.9 .* (.-yc ./ maximum(-yc)) .* 1e-3
η    = CuArray(repeat(η_y, 1, nx)')

Pf, Tf       = simulate_real_pressure!(Pf, Tf, inj_points, time; T_inj = T_INJ_FLUID)   # Basel pressure data
Tfold        = @ones(nx,ny) .* (Tf)
Trold        = @ones(nx,ny) .* (Tr)
Pold         = @ones(nx,ny) .* (Pf)
kc0          = assign_P0(kc0, inj_points, 1e-11)     # high permeability at the injection points
kc           = kc0 .* exp.(-abs.(σn) .* _σ_star .* (dt_eff./86400))
k_ini        = copy(kc)
a            = sqrt.(12 .* kc)                       # hydraulic aperture (cubic law)
ϕ            = @. ϕ_rock * (kc / k_rock)^0.2
ϕ0           = copy(ϕ)
k_η          = @. kc / η
_βϕ          = @. _β / ϕ
k_η_βϕ       = @. k_η * _βϕ
λ_ρCp_ϕ_f    = @. λ_ρCp_f / ϕ
λ_ρCp_1ϕ_r   = @. λ_ρCp_r / (1 - ϕ)
_ϕρCp_fluid  = @. 1 / ρCp_fluid / ϕ
_1ϕρCp_rock  = @. 1 / ρCp_rock / (1 - ϕ)

# =============================================================================
#  Output time series and event detection
# =============================================================================
nt         = Int(ceil(ttot / dt_eff))
theal      = @zeros(nx, ny)                  # time since last failure (healing)
trig_tog   = @zeros(nx, ny)                  # 1 if the cell failed this step
count_trig = 0
total_trig = zeros(1, nt)                    # number of failing cells per step
F_dev      = zeros(nt)                       # yield function at MONITOR_CELL
time_vec   = zeros(nt)
ε2_vec     = zeros(nt)                       # ε̇2_dev at MONITOR_CELL
Qinj_vec   = zeros(nt)                       # injected flow rate
P_well     = zeros(nt)                       # fluid pressure at MONITOR_WELL
W_el_vec          = zeros(nt)   # instantaneous total elastic energy      [J/m]
E_pl_seismic_vec  = zeros(nt)   # plastic dissipation in seismic cells    [J/m]
E_pl_aseismic_vec = zeros(nt)   # plastic dissipation in aseismic cells   [J/m]
E_elastic_vec     = zeros(nt)   # elastic work increment                  [J/m]
ε2_total_vec      = zeros(nt)   # domain-integrated deviatoric strain
ε2_seismic_vec    = zeros(nt)   # same, seismic cells only
W_pl_cumul          = @zeros(nx, ny)   # cumulative plastic dissipation map [J/m]
W_pl_aseismic_cumul = @zeros(nx, ny)   # cumulative aseismic dissipation map [J/m]

event_state          = Dict{Int,Bool}()
event_catalog        = Dict{Int,Dict}()
event_U0_store       = Dict{Int, Dict{CartesianIndex{2}, SVector{2,Float64}}}()   # displacement at event start
cell_to_event        = Dict{CartesianIndex{2}, Int}()
next_event_id_ref    = Ref(1)
Mw_list     = Float64[]
Mw_time     = Float64[]
Mw_location = Tuple{Float64,Float64}[]
L_patch     = Float64[]

# =============================================================================
#  Output helpers (file names are kept identical to the original outputs)
# =============================================================================
itag(it) = lpad(round(Int, it/1e4), 6, '0')
write_field(name, A) = CSV.write(joinpath(OUT_DIR, name * ".csv"), Tables.table(Array(A)))

function write_catalog()
    Mw_x = [loc[1] for loc in Mw_location]
    Mw_y = [loc[2] for loc in Mw_location]
    CSV.write(joinpath(OUT_DIR, "Mw_catalog.csv"),
              DataFrame(time = Mw_time, Mw = Mw_list, x = Mw_x, y = Mw_y, L_rupture = L_patch))
end

function write_timeseries(n)
    df = DataFrame(time          = time_vec[1:n],
                   strain        = ε2_vec[1:n],
                   F             = F_dev[1:n],
                   Qinj          = Qinj_vec[1:n],
                   trig          = total_trig[1:n],
                   W_elastic     = W_el_vec[1:n],
                   E_pl_seismic  = E_pl_seismic_vec[1:n],
                   E_pl_aseismic = E_pl_aseismic_vec[1:n],
                   E_elastic     = E_elastic_vec[1:n],
                   e2_tot        = ε2_total_vec[1:n],
                   e2_seismic    = ε2_seismic_vec[1:n],
                   Pwell         = P_well[1:n])
    CSV.write(joinpath(OUT_DIR, "inj_data.csv"), df)
end

burst_dir = joinpath(OUT_DIR, "burst_saves")
SAVE_BURST && mkpath(burst_dir)

# =============================================================================
#  Time loop
# =============================================================================
it = 0; ittot = 0
while it < nt
    global time, dt, Pf, Tf, it, ittot, count_trig, damp, seismic_mask

    # time step (diffusion / mechanics / advection limited)
    dt = it == 0 ? dt_eff :
         min(dt_eff, adv_lim * min(dx/maximum(abs.(qDx2)), dy/maximum(abs.(qDy2))) / 2.1)
    time += dt
    _dt   = 1/dt

    # pseudo-transient parameters, temperature
    _ReTf     = 1 / (π + sqrt(π^2 + max_lxy^2 / maximum(λ_ρCp_ϕ_f) * _dt))
    θTf_dτ    = max_lxy * _ReTf / Vpdτ
    _1_θTf_dτ = 1 / (1.0 + θTf_dτ)
    _ReTr     = 1 / (π + sqrt(π^2 + max_lxy^2 / maximum(λ_ρCp_1ϕ_r) * _dt))
    θTr_dτ    = max_lxy * _ReTr / Vpdτ
    _1_θTr_dτ = 1 / (1.0 + θTr_dτ)
    @parallel compute_iter_params_T!(dτ_ρTf, dτ_ρTr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, _ReTf, _ReTr, Vpdτ, max_lxy)

    # pseudo-transient parameters, pressure
    a       .= sqrt.(12 .* kc)
    _Re      = 1 / (π + sqrt(π^2 + max_lxy^2 / maximum(k_η_βϕ) * _dt))
    θr_dτ    = max_lxy * _Re / Vpdτ
    _1_θr_dτ = 1 / (1.0 + θr_dτ)
    @parallel compute_iter_params_P!(dτ_ρ, k_η_βϕ, _Re, Vpdτ, max_lxy)

    # ---- coupled TH pseudo-transient iterations -----------------------------
    iter = 0; errP = 2 * tol; errTf = 2 * tol; errTr = 2 * tol
    while max(errP, errTf, errTr) > tol && iter < itMax
        # hydraulics
        @parallel compute_Darcyflux!(qDx, qDy, qDx2, qDy2, Pf, Tf, k_η, θr_dτ, _1_θr_dτ, _dx, _dy, αρgy)
        @parallel update_Pf!(Pf, Pold, qDx, qDy, dτ_ρ, _dt, _dx, _dy, _βϕ)
        @parallel flux_velocity(v, qDx2, qDy2, ϕ)
        @assert !any(isnan, Pf) "NaN in Pf after solve"

        # thermics
        @parallel compute_Tfrictional!(T_frictional, βT_ρCp, ε2_pl, Tau2)
        @parallel compute_Tflux!(qTfx, qTfy, qTrx, qTry, qTfx2, qTfy2, qTrx2, qTry2, Tf, Tr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, θTf_dτ, θTr_dτ, _1_θTf_dτ, _1_θTr_dτ, _dx, _dy)
        @parallel advect_T!(dTdt, Tf, qDx2, qDy2, _dx, _dy, ϕ)
        @parallel T_transfer!(Q, h, a, dx, Tf, Tr, v)
        @parallel update_T!(Tf, Tr, Tfold, Trold, qTfx, qTfy, qTrx, qTry, dTdt, _dt, dτ_ρTf, dτ_ρTr, T_frictional, Q, ϕ, _ϕρCp_fluid, _1ϕρCp_rock, _dx, _dy)
        @parallel (1:size(Tf, 2)) bc_x!(Tf)
        @parallel (1:size(Tr, 2)) bc_x!(Tr)

        iter += 1
        if iter % nout == 0
            @parallel check_res!(ResP, ResTf, ResTr, Pf, Pold, qDx2, qDy2, Tf, Tr, Tfold, Trold, dTdt, T_frictional, qTfx2, qTfy2, qTrx2, qTry2, _dt, _dx, _dy, _βϕ, Q, _ϕρCp_fluid, _1ϕρCp_rock, ϕ)
            CUDA.has_cuda() && CUDA.synchronize()
            errP  = norm(ResP)  / sqrt(length(ResP))
            errTf = norm(ResTf) / sqrt(length(ResTf))
            errTr = norm(ResTr) / sqrt(length(ResTr))
            isnan(errP)  && error("NaN Pf")
            isnan(errTr) && error("NaN Tr")
            isnan(errTf) && error("NaN Tf")
        end
    end

    # injection boundary condition: measured Basel pressure
    Pf, Tf = simulate_real_pressure!(Pf, Tf, inj_points, time; T_inj = T_INJ_FLUID)

    # ---- mechanics ------------------------------------------------------------
    # total stress (poro- and thermo-elastic contributions)
    @parallel total_stress!(σxx, σyy, τxy, σxx_tot, σyy_tot, τxy_tot, Pf, P_hydro, Tr, Trold, K, α_rock)
    @parallel (1:size(σxx_tot, 2)) bc_x!(σxx_tot); @parallel (1:size(σyy_tot, 2)) bc_x!(σyy_tot); @parallel (1:size(τxy_tot, 2)) bc_x!(τxy_tot)
    @parallel (1:size(σxx_tot, 1)) bc_y!(σxx_tot); @parallel (1:size(σyy_tot, 1)) bc_y!(σyy_tot); @parallel (1:size(τxy_tot, 1)) bc_y!(τxy_tot)
    σyy_tot[:, end] .= bc_y      # fixed stresses on top boundary
    σxx_tot[:, end] .= bc_x
    τxy_tot[:, end] .= bc_xy
    @parallel initial_stress!(σxx, σyy, τxy, σxx_tot, σyy_tot, τxy_tot, σxx_dev, σyy_dev, τxy_dev, P_hydro, P0, Pf)

    # velocity update and velocity boundary conditions
    @parallel velocity_update(σxx_tot, σyy_tot, τxy_tot, Vx, Vy, _dx, _dy, dt, damp, ρ_rock, _ρ_elastic, σxx_dx, σyy_dy, τxy_dx, τxy_dy)
    Vx[1,:]   .= 0    # left
    Vx[end,:] .= 0    # right
    Vx[:,1]   .= 0    # bottom
    Vy[:,1]   .= 0    # bottom
    Vy[:,end] .= 0    # top
    @parallel update_displacement!(Ux, Uy, Vx, Vy, dt)

    # strain rates
    compute_strainrate_tensor!(εxx, εyy, εxy, Vx, Vy, _dx, _dy)
    @parallel compute_deviatoric_strainrate!(εxx_dev, εyy_dev, εxy_dev, εxx, εyy, εxy, divergence, ε2_dev)

    # elastic predictor
    @parallel compute_stress_predictor!(σxx_check, σyy_check, τxy_check, σxx_dev, σyy_dev, τxy_dev, P_stress, P0, dt, G, K, εxx_dev, εyy_dev, εxy_dev, divergence)

    # plastic corrector: fractures (frictional, orientation-dependent) then matrix (Drucker-Prager).
    # ε2_pl is a plastic strain RATE [1/s] in both matrix and fractures
    # (consistent with ε2_dev, dE_cell and T_frictional).
    @parallel plastic_corrector_faults_2D!(σxx_check, σyy_check, τxy_check, P_stress, σxx_dev, σyy_dev, τxy_dev, G, K, Tau2, F_check, ε2_pl, F_fail, F_type, material_d, dip_d, coh_f, phi_f, psi_f, tensile, dt)
    @parallel plastic_corrector_matrix!(σxx_check, σyy_check, τxy_check, P_stress, σxx_dev, σyy_dev, τxy_dev, Tau2, F_check, ε2_pl, F_fail, F_type, material_d, coh, friction_angle, ψ, G, K, tensile, dt)
    CUDA.has_cuda() && CUDA.synchronize()
    ε2_el .= ε2_dev .- ε2_pl
    plastic_mask = ε2_pl .> 0
    seismic_mask = (ε2_dev .- ε2_old) .> seismic_tr

    if maximum(F_fail) > 0
        damp = 0.08
    end

    # consistency check: the yield function must not stay positive after correction.
    # If it does, dump the current state for debugging and stop.
    if maximum(F_check ./ (abs.(P_stress) .+ 1e6)) > 1e-4
        write_catalog(); write_timeseries(it)
        write_field("Pf_it$(itag(it))", Pf); write_field("kc_it$(itag(it))", kc); write_field("Tf_it$(itag(it))", Tf)
        write_field("P_stress", P_stress); write_field("F_check", F_check); write_field("F_type", F_type)
        write_field("Tau2", Tau2); write_field("Vy", Vy); write_field("Sxx", σxx_tot); write_field("Syy", σyy_tot)
        write_field("W_pl_cumul_it$(itag(it))", W_pl_cumul); write_field("W_pl_aseismic_it$(itag(it))", W_pl_aseismic_cumul)
        println("Number of cells with F > 1e-6: ", sum(F_check .> 1e-6))
        println("Positions: ", findall(F_check .> 1e-6))
        error("F_check is positive")
    end

    # stresses resolved on fractures and effective stresses
    @parallel stress_on_faults_2d!(σxx_dev, σyy_dev, τxy_dev, P_stress, σn, τ, dip_d)
    @parallel compute_effective_stress!(σxx_dev, σyy_dev, τxy_dev, σxx, σyy, τxy, P_stress)
    @assert !any(isnan, σxx) "NaN in σxx"
    @assert !any(isnan, σyy) "NaN in σyy"
    @assert !any(isnan, τxy) "NaN in τxy"

    # ---- seismic event detection (after DETECT_AFTER seconds) -------------------
    if time > DETECT_AFTER
        # angle_2d and material are the CPU copies of the orientation and material maps
        detect_seismic_events_fast!(ε2_dev, ε2_old, Vx, Vy, Ux, Uy, angle_2d, material,
            event_state, event_catalog, event_U0_store, cell_to_event, time, dt, dx, dy, G,
            Mw_list, Mw_time, Mw_location, L_patch, next_event_id_ref, xc, yc;
            threshold    = seismic_tr,
            max_duration = 20.0)       # [s]
    end

    # ---- energy budget ------------------------------------------------------------
    dE_cell = Tau2 .* ε2_pl .* dt .* dx .* dy      # plastic dissipation per cell [J/m]
    W_el_vec[it+1]          = sum(Tau2.^2 ./ (2G)) .* dx .* dy
    E_pl_seismic_vec[it+1]  = sum(dE_cell[ seismic_mask .& plastic_mask])
    E_pl_aseismic_vec[it+1] = sum(dE_cell[.!seismic_mask .& plastic_mask])
    E_elastic_vec[it+1]     = sum((Tau2 .* ε2_el)) .* dt .* dx .* dy
    ε2_total_vec[it+1]      = sum(ε2_dev) * dx * dy * dt
    ε2_seismic_vec[it+1]    = sum(ε2_dev[seismic_mask]) * dx * dy * dt
    W_pl_cumul          .+= dE_cell
    W_pl_aseismic_cumul .+= dE_cell .* (.!seismic_mask .& plastic_mask)

    # ---- permeability / porosity update ------------------------------------------
    @parallel update_permeability!(F_fail, trig_tog, σn, theal, kc, kc0, k_ini, α_heal, _σ_star, dt)
    @assert all(kc .> 0) "Non-positive permeability!"
    @parallel update_porosity!(ϕ, ϕ0, kc, k_ini, k_η, k_η_βϕ, η, _βϕ, _β, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, λ_ρCp_f, λ_ρCp_r, _ϕρCp_fluid, ρCp_fluid, _1ϕρCp_rock, ρCp_rock)

    # ---- time series ------------------------------------------------------------------
    count_trig       = sum(trig_tog)
    total_trig[it+1] = count_trig
    Qinj_vec[it+1]   = compute_total_flow(qDx2, qDy2, inj_points, dx, dy)
    F_dev[it+1]      = @allowscalar F_check[MONITOR_CELL...]
    ε2_vec[it+1]     = @allowscalar ε2_dev[MONITOR_CELL...]
    P_well[it+1]     = @allowscalar Pf[MONITOR_WELL...]
    time_vec[it+1]   = time

    ittot += iter; it += 1
    count_trig = 0
    @parallel assign!(Pold, Pf)
    @parallel assign!(Tfold, Tf)
    @parallel assign!(Trold, Tr)
    @parallel assign!(P0, P_stress)
    @parallel assign!(ε2_old, ε2_dev)

    # ---- outputs ------------------------------------------------------------------------
    # high-frequency snapshots around seismicity onset (Coulomb stress analysis)
    if SAVE_BURST && BURST_WINDOW[1] <= it <= BURST_WINDOW[2] && it % BURST_EVERY == 0
        idx = lpad(it, 8, '0')
        for (name, A) in (("Pf", Pf), ("S_normal", σn), ("Tau", τ), ("Pstress", P_stress),
                          ("Tau2", Tau2), ("Sxx", σxx_tot), ("Syy", σyy_tot))
            CSV.write(joinpath(burst_dir, "$(name)_it$(idx).csv"), Tables.table(Array(A)))
        end
    end

    # regular snapshots
    if it % SNAP_EVERY == 0
        write_catalog(); write_timeseries(it)
        tag = itag(it)
        write_field("Pf_it$tag", Pf);       write_field("kc_it$tag", kc);        write_field("kc0_it$tag", kc0)
        write_field("Tf_it$tag", Tf);       write_field("Ffail_it$tag", F_fail); write_field("F_type_it$tag", F_type)
        write_field("Tau2_it$tag", Tau2);   write_field("Tau_it$tag", τ);        write_field("S_normal_it$tag", σn)
        write_field("P_stress_it$tag", P_stress)
        write_field("plastic", ε2_pl);      write_field("dev_strain", ε2_dev);   write_field("F_check", F_check)
        write_field("Vy", Vy);              write_field("Sxx", σxx_tot);         write_field("Syy", σyy_tot)
        write_field("W_pl_cumul_it$tag", W_pl_cumul)
        write_field("W_pl_aseismic_it$tag", W_pl_aseismic_cumul)
    end
end

# =============================================================================
#  Final outputs
# =============================================================================
@printf("Total time = %1.2f, time steps = %d, nx = %d, iterations tot = %d \n", round(ttot, sigdigits=2), it, nx, ittot)
write_catalog(); write_timeseries(it)
write_field("Pf_end", Pf);             write_field("kc_end", kc);          write_field("Tf_end", Tf)
write_field("Tau2_end", Tau2);         write_field("P_stress_end", P_stress)
write_field("Tau_end", τ);             write_field("S_normal_end", σn);    write_field("plastic_end", ε2_pl)
write_field("dev_strain", ε2_dev);     write_field("Vy", Vy)
write_field("Sxx", σxx_tot);           write_field("Syy", σyy_tot);        write_field("S_normal", σn)
write_field("W_pl_cumul_it$(itag(it))", W_pl_cumul)
write_field("W_pl_aseismic_it$(itag(it))", W_pl_aseismic_cumul)
