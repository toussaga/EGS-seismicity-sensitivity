# =============================================================================
#  benchmark_lauwerier_straight.jl
#
#  Straight, "infinite" horizontal fracture in a near-impermeable matrix —
#  built specifically to match Lauwerier's (1955) assumptions.
# =============================================================================

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using SpecialFunctions

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
end

using Printf, LinearAlgebra, Plots, Statistics

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

# ── Kernels (identical to the validated thermohydraulic benchmark) ───────────
@parallel function compute_iter_params_P!(dτ_ρ, k_η_βϕ, _Re, Vpdτ, max_lxy)
    @all(dτ_ρ) = Vpdτ * max_lxy * _Re / @maxloc(k_η_βϕ)
    return
end

@parallel function compute_Darcyflux!(qDx, qDy, qDx2, qDy2, Pf, Tf,
                                       k_η, θr_dτ, _1_θr_dτ, _dx, _dy, αρgy)
    @all(qDx)  = (@all(qDx)  * θr_dτ - @av_xi(k_η) * (@d_xi(Pf) * _dx                    )) * _1_θr_dτ
    @all(qDy)  = (@all(qDy)  * θr_dτ - @av_yi(k_η) * (@d_yi(Pf) * _dy - αρgy*@av_yi(Tf)  )) * _1_θr_dτ
    @all(qDx2) = -@av_xi(k_η) * (@d_xi(Pf) * _dx)
    @all(qDy2) = -@av_yi(k_η) * (@d_yi(Pf) * _dy - αρgy * @av_yi(Tf))
    return
end

@parallel function update_Pf!(Pf, Pold, qDx, qDy, dτ_ρ, _dt, _dx, _dy, _βϕ)
    @inn(Pf) = (@inn(Pf) + @all(dτ_ρ) * (
        @inn(Pold) * _dt -
        (@d_xa(qDx) * _dx * @inn(_βϕ) + @d_ya(qDy) * _dy * @inn(_βϕ))
    )) / (1.0 + @all(dτ_ρ) * _dt)
    return
end

@parallel function flux_velocity(v, qDx2, qDy2, ϕ)
    @all(v) = sqrt((@av_xa(qDx2) / @inn(ϕ))^2 + (@av_ya(qDy2) / @inn(ϕ))^2)
    return
end

@parallel function compute_iter_params_T!(dτ_ρTf, dτ_ρTr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, _ReTf, _ReTr, Vpdτ, max_lxy)
    @all(dτ_ρTf)  = Vpdτ * max_lxy * _ReTf / @maxloc(λ_ρCp_ϕ_f)
    @all(dτ_ρTr)  = Vpdτ * max_lxy * _ReTr / @maxloc(λ_ρCp_1ϕ_r)
    return
end

@parallel function compute_Tfrictional!(T_frictional, βT_ρCp, ϵ, τ_heat)
    @inn(T_frictional) = βT_ρCp * @inn(ϵ) * @all(τ_heat)
    return
end

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

@parallel_indices (ix,iy) function advect_T!(dTdt, T, qDx2, qDy2, _dx, _dy, ϕ)
    if (ix<=size(dTdt, 1) && iy<=size(dTdt, 2))
        dTdt[ix, iy] = ((qDx2[ix, iy]   > 0)  * qDx2[ix, iy]   * (T[ix+1, iy+1] - T[ix, iy+1])   * _dx +
                        (qDx2[ix+1, iy] < 0)  * qDx2[ix+1, iy] * (T[ix+2, iy+1] - T[ix+1, iy+1]) * _dx +
                        (qDy2[ix, iy]   > 0)  * qDy2[ix, iy]   * (T[ix+1, iy+1] - T[ix+1, iy])   * _dy +
                        (qDy2[ix, iy+1] < 0)  * qDy2[ix, iy+1] * (T[ix+1, iy+2] - T[ix+1, iy+1]) * _dy) / @inn(ϕ)
    end
    return
end

@parallel function T_transfer!(Q, h, a, dx, Tf, Tr, v)
    @all(h) = 0.00905 * ((0.0009217 + exp(-2.314e5 * @inn(a)))^(-1) * (0.0693 + exp(-136.9 * @all(v)))^(-1)) + 650 * @all(v)
    @all(h) = 50000
    @all(Q) = @all(h) * 2 * dx * (@inn(Tf) - @inn(Tr))
    return
end

@parallel function update_T!(Tf, Tr, Tfold, Trold, qTfx, qTfy, qTrx, qTry, dTdt, _dt, dτ_ρTf, dτ_ρTr, T_frictional, Q, ϕ, _ϕρCp_fluid, _1ϕρCp_rock, _dx, _dy)
    @inn(Tf) = (@inn(Tf) +  @all(dτ_ρTf) * (@inn(Tfold) * _dt - (@d_xa(qTfx) * _dx + @d_ya(qTfy) * _dy) - @all(Q) * @inn(_ϕρCp_fluid) - @all(dTdt)                        )) / (1.0 + @all(dτ_ρTf) * _dt)
    @inn(Tr) = (@inn(Tr) +  @all(dτ_ρTr) * (@inn(Trold) * _dt - (@d_xa(qTrx) * _dx + @d_ya(qTry) * _dy) + @all(Q) * @inn(_1ϕρCp_rock) + @inn(T_frictional) / (1 - @inn(ϕ)))) / (1.0 + @all(dτ_ρTr) * _dt)
    return
end

@parallel function check_res_P!(ResP, Pf, Pold, qDx2, qDy2, _dt, _dx, _dy, _βϕ)
    @all(ResP)  = -(@inn(Pf) - @inn(Pold))  * _dt - (@d_xa(qDx2) * _dx  * @inn(_βϕ) + @d_ya(qDy2) * _dy * @inn(_βϕ))
    return
end

@parallel function check_res!(ResP, ResTf, ResTr, Pf, Pold, qDx2, qDy2, Tf, Tr, Tfold, Trold, dTdt, T_frictional, qTfx2, qTfy2, qTrx2, qTry2, _dt, _dx, _dy, _βϕ, Q, _ϕρCp_fluid, _1ϕρCp_rock, ϕ)
    @all(ResP)  = -(@inn(Pf) - @inn(Pold))  * _dt - (@d_xa(qDx2) * _dx  * @inn(_βϕ) + @d_ya(qDy2) * _dy * @inn(_βϕ))
    @all(ResTf) = -(@inn(Tf) - @inn(Tfold)) * _dt - (@d_xa(qTfx2) * _dx             + @d_ya(qTfy2) * _dy           ) - @all(Q) * @inn(_ϕρCp_fluid) - @all(dTdt)
    @all(ResTr) = -(@inn(Tr) - @inn(Trold)) * _dt - (@d_xa(qTrx2) * _dx             + @d_ya(qTry2) * _dy           ) + @all(Q) * @inn(_1ϕρCp_rock) + @inn(T_frictional) / (1 - @inn(ϕ))
    return
end

# ── Boundary conditions ──────────────────────────────────────────────────────
@parallel_indices (iy) function bc_dirichlet_x!(A, v_left, v_right)
    A[1,   iy] = v_left
    A[end, iy] = v_right
    return
end

@parallel_indices (ix) function bc_neumann_y!(A)
    A[ix, 1  ] = A[ix, 2    ]
    A[ix, end] = A[ix, end-1]
    return
end

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

# ── Analytical solution — Lauwerier (1955), transient ─────────────────────────
function lauwerier_temperature(x, t, u, λ_m, ρCp_m, ρCp_f, b_half, T0, Tinj)
    t <= x/u && return T0
    ω = sqrt(λ_m * ρCp_m) / (b_half * ρCp_f * u)
    ξ = ω * x / (2*sqrt(t - x/u))
    return T0 + (Tinj - T0) * erfc(ξ)
end

# ===========================================================================
#  PARAMETERS
# ===========================================================================
# Domain: long in x (fracture direction), just tall enough in y to resolve
# matrix conduction a few penetration depths out (which is CENTIMETERS —
# κ_rock ~ 8e-7 m^2/s, so sqrt(4κt) over ~1000s is only a few cm).
lx, ly   = 20.0, 1.0
nx, ny   = 4801, 101

b        = 0.05                      # fracture aperture [m]
aperture = b
K_m      = 1e-14                     # matrix permeability — genuinely impermeable, not just "low"
K_f      = 1e-10                     # fracture permeability — same order as the ex6 case
mu_fluid = 1e-3                      # fluid viscosity [Pa.s]

ρ_fluid    = 1000.0
ρ_rock     = 2700.0
ϕ_rock     = 0.03
β_fluid    = 1e-10
β_phi      = 1e-8
λ_rock     = 2.1
λ_fluid    = 0.6
Cp_fluid   = 4200.0
Cp_rock    = 1000.0
βT         = 1.0
α          = 0.0                     # buoyancy off — same idealization as the thermohydraulic benchmark
gx, gy     = 0.0, 0.0

ρCp_fluid  = ρ_fluid * Cp_fluid
ρCp_rock   = ρ_rock  * Cp_rock
λ_ρCp_f    = λ_fluid / ρCp_fluid
λ_ρCp_r    = λ_rock  / ρCp_rock
_β         = 1 / (β_fluid + β_phi)
ρCp_ratio  = ρCp_fluid / ρCp_rock
βT_ρCp     = βT / ρCp_rock

# Hydraulic BC — same pressure GRADIENT as the ex6 case (ΔP/lx), scaled to
# the longer domain, so u_frac ends up a similar order of magnitude.
dPdx_ref = (2*4.9646e5) / 10.0
P_left   =  0.5 * dPdx_ref * lx
P_right  = -0.5 * dPdx_ref * lx

# Thermal BC
T_right  = 353.15      # background/reservoir temperature [K] (80 degC)
T_left   = 293.15      # injected fluid temperature [K]       (20 degC)

ε2_pl_val = 0.0
Tau2_val  = 0.0

tol      = 1e-8
itMax    = 30_000
nout     = 50
noutT    = 50
CFL      = 1 / sqrt(2)

dx, dy   = lx / nx, ly / ny
_dx, _dy = 1/dx, 1/dy
xn = LinRange(0, lx, nx+1);  xc = av1(xn)
yn = LinRange(0, ly, ny+1);  yc = av1(yn)
max_lxy  = max(lx, ly)
Vpdτ     = CFL * min(dx, dy)
dt       = 1e30
_dt      = 1 / dt

# ===========================================================================
#  PERMEABILITY FIELD — horizontal fracture spanning the FULL domain length
# ===========================================================================
y_frac  = ly / 2
K_cpu   = fill(K_m, nx, ny)
frac_j  = findall(j -> abs(yc[j] - y_frac) <= b/2, 1:ny)
K_cpu[:, frac_j] .= K_f
j_frac  = frac_j[argmin(abs.(yc[frac_j] .- y_frac))]   # row closest to fracture centerline, for plotting

@printf("\n── Geometry check ────────────────────────────────────────────\n")
@printf("  Fracture rows (y-index): %s  (y = %s)\n", frac_j, round.(yc[frac_j], digits=4))
@printf("  dx = %.5f m, dy = %.5f m, aperture b = %.3f m\n", dx, dy, b)
@printf("──────────────────────────────────────────────────────────────\n\n")

K_field  = Data.Array(K_cpu)
ϕ        = @zeros(nx, ny) .+ ϕ_rock
a_field  = @zeros(nx, ny) .+ aperture

# ===========================================================================
#  ARRAYS AND PT SETUP
# ===========================================================================
Pf         = @zeros(nx, ny)
Pold       = @zeros(nx, ny)
qDx, qDy   = @zeros(nx-1, ny-2), @zeros(nx-2, ny-1)
qDx2, qDy2 = @zeros(nx-1, ny-2), @zeros(nx-2, ny-1)
ResP       = @zeros(nx-2, ny-2)
dτ_ρ       = @zeros(nx-2, ny-2)
Tf_dummy   = @zeros(nx, ny)
v          = @zeros(nx-2, ny-2)
h          = @zeros(nx-2, ny-2)

Tf, Tr         = @zeros(nx, ny), @zeros(nx, ny)
qTfx, qTfy     = @zeros(nx-1, ny-2), @zeros(nx-2, ny-1)
qTfx2, qTfy2   = @zeros(nx-1, ny-2), @zeros(nx-2, ny-1)
qTrx, qTry     = @zeros(nx-1, ny-2), @zeros(nx-2, ny-1)
qTrx2, qTry2   = @zeros(nx-1, ny-2), @zeros(nx-2, ny-1)
dTdt           = @zeros(nx-2, ny-2)
T_frictional   = @zeros(nx, ny)
ResTf          = @zeros(nx-2, ny-2)
ResTr          = @zeros(nx-2, ny-2)
Q              = @zeros(nx-2, ny-2)
dτ_ρTf, dτ_ρTr = @zeros(nx-2, ny-2), @zeros(nx-2, ny-2)

# NOTE: no depth-dependent viscosity here — uniform mu_fluid. This is an
# idealized 1D-in-spirit validation case, not the reservoir-depth geometry.
k_η          = K_field ./ mu_fluid
_βϕ          = @. _β / ϕ
k_η_βϕ       = @. k_η * _βϕ
λ_ρCp_ϕ_f    = @. λ_ρCp_f / ϕ
λ_ρCp_1ϕ_r   = @. λ_ρCp_r / (1 - ϕ)
_ϕρCp_fluid  = @. 1 / ρCp_fluid / ϕ
_1ϕρCp_rock  = @. 1 / ρCp_rock / (1 - ϕ)

_Re        = 1 / (pi + sqrt(pi^2 + max_lxy^2 / maximum(k_η_βϕ) * _dt))
theta_r    = max_lxy * _Re / Vpdτ
_1_theta_r = 1 / (1.0 + theta_r)
@parallel compute_iter_params_P!(dτ_ρ, k_η_βϕ, _Re, Vpdτ, max_lxy)

@parallel (1:ny) bc_dirichlet_x!(Pf, P_left, P_right)
@parallel (1:nx) bc_neumann_y!(Pf)

ε2_pl = fill(ε2_pl_val, nx, ny)
Tau2  = fill(Tau2_val,  nx-2, ny-2)

Tf .= T_right
Tr .= T_right
Tfold = copy(Tf)
Trold = copy(Tr)
@parallel (1:ny) bc_dirichlet_x!(Tf, T_left, T_right)

# ===========================================================================
#  PRESSURE — STEADY-STATE PSEUDO-TRANSIENT LOOP
# ===========================================================================
@printf("Starting pressure PT iteration  (tol=%.1e, itMax=%d)\n", tol, itMax)
iter = 0; errP = 2*tol

while errP > tol && iter < itMax
    global iter; global errP
    @parallel compute_Darcyflux!(qDx, qDy, qDx2, qDy2, Pf, Tf_dummy, k_η, theta_r, _1_theta_r, _dx, _dy, 0.0)
    @parallel update_Pf!(Pf, Pold, qDx, qDy, dτ_ρ, _dt, _dx, _dy, _βϕ)
    @parallel flux_velocity(v, qDx2, qDy2, ϕ)

    @parallel (1:ny) bc_dirichlet_x!(Pf, P_left, P_right)
    @parallel (1:nx) bc_neumann_y!(Pf)

    iter += 1
    if iter % nout == 0
        @parallel check_res_P!(ResP, Pf, Pold, qDx2, qDy2, _dt, _dx, _dy, _βϕ)
        errP = norm(ResP) / sqrt(length(ResP))
        @printf("  iter = %5d  |  errP = %.3e\n", iter, errP)
    end
end
errP <= tol ? (@printf("Pressure converged in %d iterations (errP = %.2e)\n", iter, errP)) :
              (@printf("WARNING: pressure did not converge (errP = %.2e)\n", errP))

v_cpu   = Array(v)
u_frac  = mean(v_cpu[K_cpu[2:end-1, 2:end-1] .== K_f])
@printf("\nFracture velocity (converged numerical solve): %.4e m/s\n", u_frac)
@printf("Matrix velocity (should be ~0):                 %.4e m/s\n", mean(v_cpu[K_cpu[2:end-1, 2:end-1] .== K_m]))

# ===========================================================================
#  TEMPERATURE — TRANSIENT
# ===========================================================================
t_cross_domain = lx / u_frac
target_frac    = 0.4                    # stop well before the front reaches the outlet
t_end          = target_frac * t_cross_domain

qDx2_cpu, qDy2_cpu = Array(qDx2), Array(qDy2)
adv_lim  = ϕ_rock / ρCp_ratio
dt_adv   = adv_lim * min(dx/maximum(abs.(qDx2_cpu)), dy/maximum(abs.(qDy2_cpu))) / 2.1
dt_phys  = dt_adv
n_tsteps = ceil(Int, t_end / dt_phys)
snapshot_steps = clamp.(round.(Int, [0.2, 0.4, 0.6, 0.8, 1.0] .* n_tsteps), 1, n_tsteps)

_dt_T = 1 / dt_phys
_ReTf = 1 / (pi + sqrt(pi^2 + max_lxy^2 / maximum(λ_ρCp_ϕ_f)  * _dt_T))
_ReTr = 1 / (pi + sqrt(pi^2 + max_lxy^2 / maximum(λ_ρCp_1ϕ_r) * _dt_T))
θTf_dτ, θTr_dτ = max_lxy*_ReTf/Vpdτ, max_lxy*_ReTr/Vpdτ
_1_θTf_dτ, _1_θTr_dτ = 1/(1.0+θTf_dτ), 1/(1.0+θTr_dτ)
@parallel compute_iter_params_T!(dτ_ρTf, dτ_ρTr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, _ReTf, _ReTr, Vpdτ, max_lxy)

@printf("\nTransient thermal march: dt_phys=%.3e s, %d steps (t_end=%.3e s, %.0f%% of domain crossing)\n",
        dt_phys, n_tsteps, t_end, 100*target_frac)

t_phys = 0.0
snapshots = Dict{Int, Tuple{Float64,Vector{Float64}}}()

for it_t in 1:n_tsteps
    global t_phys
    Tfold .= Tf
    Trold .= Tr

    iterT = 0; errT = 2*tol
    while errT > tol && iterT < itMax
        @parallel compute_Tfrictional!(T_frictional, βT_ρCp, ε2_pl, Tau2)
        @parallel compute_Tflux!(qTfx, qTfy, qTrx, qTry, qTfx2, qTfy2, qTrx2, qTry2, Tf, Tr, λ_ρCp_ϕ_f, λ_ρCp_1ϕ_r, θTf_dτ, θTr_dτ, _1_θTf_dτ, _1_θTr_dτ, _dx, _dy)
        @parallel advect_T!(dTdt, Tf, qDx2, qDy2, _dx, _dy, ϕ)
        @parallel T_transfer!(Q, h, a_field, dx, Tf, Tr, v)
        @parallel update_T!(Tf, Tr, Tfold, Trold, qTfx, qTfy, qTrx, qTry, dTdt, _dt_T, dτ_ρTf, dτ_ρTr, T_frictional, Q, ϕ, _ϕρCp_fluid, _1ϕρCp_rock, _dx, _dy)

        @parallel (1:ny) bc_dirichlet_x!(Tf, T_left, T_right)
        @parallel (1:nx) bc_neumann_y!(Tf)
        @parallel (1:size(Tr, 2)) bc_x!(Tr)
        @parallel (1:size(Tr, 1)) bc_y!(Tr)

        iterT += 1
        if iterT % noutT == 0
            @parallel check_res!(ResP, ResTf, ResTr, Pf, Pold, qDx2, qDy2, Tf, Tr, Tfold, Trold,
                                  dTdt, T_frictional, qTfx2, qTfy2, qTrx2, qTry2, _dt_T, _dx, _dy,
                                  _βϕ, Q, _ϕρCp_fluid, _1ϕρCp_rock, ϕ)
            errT = max(norm(ResTf)/sqrt(length(ResTf)), norm(ResTr)/sqrt(length(ResTr)))
        end
    end

    t_phys += dt_phys
    if it_t in snapshot_steps
        snapshots[it_t] = (t_phys, Array(Tf)[:, j_frac])
        @printf("  step %5d  t=%.3e s  (inner iters=%d)\n", it_t, t_phys, iterT)
    end
end

# ===========================================================================
#  COMPARISON — Lauwerier
# ===========================================================================
xc_arr = Array(xc)
b_half = aperture/2
h_frac = 0.00905 * ((0.0009217 + exp(-2.314e5*aperture))^(-1) *
                     (0.0693   + exp(-136.9*u_frac))^(-1)) + 650*u_frac

@printf("\n── Thermal benchmark vs Lauwerier (straight fracture) ──────────────────\n")
plot_data = NamedTuple[]
x_max_valid = 0.7*lx   # stay away from the outlet BC
for it_t in sort(collect(keys(snapshots)))
    t_snap, Tf_row = snapshots[it_t]
    Tf_lauw = [ x <= 0 ? T_right :
                lauwerier_temperature(x, t_snap, u_frac, λ_rock, ρCp_rock, ρCp_fluid, b_half, T_right, T_left)
                for x in xc_arr ]

    mask = (xc_arr .> 0) .& (xc_arr .< x_max_valid)
    err  = Tf_row[mask] .- Tf_lauw[mask]
    rng  = maximum(Tf_lauw[mask]) - minimum(Tf_lauw[mask])
    NRMSE = rng > 0 ? sqrt(mean(err.^2))/rng*100 : NaN
    NMAE  = rng > 0 ? mean(abs.(err))/rng*100    : NaN
    pct_crossing = t_snap / t_cross_domain * 100
    @printf("  t=%.3e s  (%5.1f%% of crossing)  NRMSE=%6.3f %%  NMAE=%6.3f %%  (h=%.3e)\n",
            t_snap, pct_crossing, NRMSE, NMAE, h_frac)

    push!(plot_data, (t=t_snap, pct=pct_crossing, NRMSE=NRMSE, NMAE=NMAE, x=xc_arr[mask], Tnum=Tf_row[mask], Tlauw=Tf_lauw[mask]))
end
@printf("──────────────────────────────────────────────────────────────────────\n")

# ===========================================================================
#  PLOTS
# ===========================================================================
plt_T = plot(xlabel="Distance from inlet [m]", ylabel="Tf [°C]",
             title="Lauwerier benchmark — Temperature along fracture",
             legend=:outertopright, framestyle=:box, size=(950,550))
cmap = palette(:viridis, length(plot_data))
for (k, d) in enumerate(plot_data)
    plot!(plt_T, d.x, d.Tlauw .- 273.15, color=cmap[k], linewidth=2, label=@sprintf("Lauwerier, t=%.0fs", d.t))
    scatter!(plt_T, d.x[1:10:end], d.Tnum[1:10:end].- 273.15, color=cmap[k], markersize=3,
             markerstrokewidth=0, label=@sprintf("Numerical, t=%.0fs", d.t))
end
#display(plt_T)
#savefig(plt_T, "benchmark_lauwerier_straight_result.png")
#@printf("Figure saved to benchmark_lauwerier_straight_result.png\n")

t_last, Tf_last_row = snapshots[maximum(keys(snapshots))]
plt_field = heatmap(xc_arr, Array(yc), Array(Tf)' .- 273.15,
    title=@sprintf("Tf field at t=%.0f s [°C]", t_last),
    xlabel="x [m]", ylabel="y [m]", aspect_ratio=:auto, c=:coolwarm, framestyle=:box)

fullfig = plot(plt_field, plt_T, layout=(1,2), size=(1400,500), dpi=300)
display(fullfig)
savefig(fullfig, "benchmark_lauwerier_straight_2.pdf")
@printf("Figure saved to benchmark_lauwerier_straight_2.png\n")
