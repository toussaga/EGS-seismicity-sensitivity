# =============================================================================
#  benchmark_hydraulic.jl
# =============================================================================

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using Statistics

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=false)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=false)
end

using Printf, LinearAlgebra, Plots

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

# ── Your kernels — unchanged ────────────────────────────────────────────────
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

@parallel function check_res!(ResP, Pf, Pold, qDx2, qDy2, _dt, _dx, _dy, _βϕ)
    @all(ResP) = -((@inn(Pf) - @inn(Pold)) * _dt +
                    @d_xa(qDx2) * _dx * @inn(_βϕ) +
                    @d_ya(qDy2) * _dy * @inn(_βϕ))
    return
end

# ── Boundary conditions ──────────────────────────────────────────────────────
@parallel_indices (iy) function bc_dirichlet_x!(Pf, P_left, P_right)
    Pf[1,   iy] = P_left
    Pf[end, iy] = P_right
    return
end

@parallel_indices (ix) function bc_neumann_y!(Pf)
    Pf[ix, 1  ] = Pf[ix, 2    ]
    Pf[ix, end] = Pf[ix, end-1]
    return
end

# ── Fracture permeability with axis clipping ─────────────────────────────────
#
#  Two conditions must BOTH be satisfied for a cell to be inside the fracture:
#
#   1. d_perp ≤ b/2      (within the aperture band, perpendicular to fracture)
#   2. d_axis ≤ L/2      (within the fracture length, along the fracture axis)
#
function build_permeability_clipped!(K_field, xc, yc, nx, ny,
                                      K_m, K_f, b, alpha_frac,
                                      x1, y1, x2, y2)
    cx      = 0.5 * (x1 + x2)
    cy      = 0.5 * (y1 + y2)
    L_half  = 0.5 * sqrt((x2-x1)^2 + (y2-y1)^2)   # half-length = 1.0 m
    half_b  = 0.5 * b

    # unit tangent (along fracture) and unit normal (perpendicular)
    tx =  cos(alpha_frac);  ty = sin(alpha_frac)
    nx_ = -sin(alpha_frac); ny_ = cos(alpha_frac)

    K_field .= K_m
    for j in 1:ny, i in 1:nx
        dx_c   = xc[i] - cx
        dy_c   = yc[j] - cy
        d_perp = abs(dx_c * nx_ + dy_c * ny_)
        d_axis = abs(dx_c * tx  + dy_c * ty )
        if d_perp <= half_b && d_axis <= L_half
            K_field[i, j] = K_f
        end
    end
end

# ── Analytical solution (complex potential, from THERMAID ex6.m) ─────────────
function analytical_pressure(xc, yc, nx, ny, alpha, b_max, L, K_m, K_f, mu, q0)
    x = xc .- (xc[1] + xc[end]) / 2
    y = yc .- (yc[1] + yc[end]) / 2

    z1     = (-1 + 0im) * exp(im * alpha)
    z2     = ( 1 + 0im) * exp(im * alpha)
    z_mid  = 0.5 * (z1 + z2)
    z_half = 0.5 * (z2 - z1)

    A = 0.5 * K_f * b_max / (K_m * L + K_f * b_max) * q0 * L * cos(alpha)

    p_ana = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        z = x[i] + im * y[j]
        Z = (z - z_mid) / z_half

        omega_pert = -A * sign(real(Z)) * sqrt((Z - 1) * (Z + 1)) + A * Z
        omega_ff   = -0.5 * q0 * L * exp(im * alpha) * Z

        p_ana[i, j] = real(omega_pert + omega_ff) * mu / K_m
    end
    return p_ana
end

# ===========================================================================
#  PARAMETERS 
# ===========================================================================
lx, ly   = 10.0, 10.0
K_m      = 1e-12
k_ratio  = 1e2
K_f      = K_m * k_ratio
b        = 0.05
mu_fluid = 1e-3
phi_val  = 0.3
beta_fl  = 1e-10
beta_phi = 1e-8
_beta    = 1 / (beta_fl + beta_phi)

P_left   =  4.9646e5
P_right  = -4.9646e5

alpha_frac = 45 * pi / 180
L_ana      = 2.0
q0_ana     = 1e-4

# Fracture endpoints — from simple_fracture.csv, analytically exact 2 m long
x1_f, y1_f = 4.291893219, 4.291893219
x2_f, y2_f = 5.707106781, 5.707106781

nx, ny   = 301, 301
tol      = 1e-8
itMax    = 10_000
nout     = 50
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
#  PERMEABILITY FIELD
# ===========================================================================
K_cpu = zeros(nx, ny)
build_permeability_clipped!(K_cpu, Array(xc), Array(yc), nx, ny,
                             K_m, K_f, b, alpha_frac,
                             x1_f, y1_f, x2_f, y2_f)

# --- diagnostic: verify discrete fracture length ---
frac_cells = findall(K_cpu .== K_f)
if !isempty(frac_cells)
    xs = [xc[ci[1]] for ci in frac_cells]
    ys = [yc[ci[2]] for ci in frac_cells]
    x_d1, x_d2 = minimum(xs), maximum(xs)
    y_d1, y_d2 = minimum(ys), maximum(ys)
    L_disc = sqrt((x_d2-x_d1)^2 + (y_d2-y_d1)^2)
    @printf("\n── Fracture geometry check ───────────────────────────────────\n")
    @printf("  Analytical tip 1 : (%.6f, %.6f)\n", x1_f, y1_f)
    @printf("  Analytical tip 2 : (%.6f, %.6f)\n", x2_f, y2_f)
    @printf("  Discrete   tip 1 : (%.6f, %.6f)\n", x_d1, y_d1)
    @printf("  Discrete   tip 2 : (%.6f, %.6f)\n", x_d2, y_d2)
    @printf("  Discrete length  : %.6f m  (target %.6f m)\n", L_disc, L_ana)
    @printf("  Tip position error tip1: %.4f m  tip2: %.4f m\n",
            sqrt((x_d1-x1_f)^2+(y_d1-y1_f)^2),
            sqrt((x_d2-x2_f)^2+(y_d2-y2_f)^2))
    @printf("  Fracture cells: %d\n", length(frac_cells))
    @printf("──────────────────────────────────────────────────────────────\n\n")
end

K_field  = Data.Array(K_cpu)
k_eta    = K_field ./ mu_fluid
_betaphi = (_beta / phi_val) .* @ones(nx, ny)
k_eta_bphi = k_eta .* _betaphi

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

_Re        = 1 / (pi + sqrt(pi^2 + max_lxy^2 / maximum(k_eta_bphi) * _dt))
theta_r    = max_lxy * _Re / Vpdτ
_1_theta_r = 1 / (1.0 + theta_r)
@parallel compute_iter_params_P!(dτ_ρ, k_eta_bphi, _Re, Vpdτ, max_lxy)

@parallel (1:ny) bc_dirichlet_x!(Pf, P_left, P_right)
@parallel (1:nx) bc_neumann_y!(Pf)

# ===========================================================================
#  PSEUDO-TRANSIENT LOOP
# ===========================================================================
@printf("Starting PT iteration  (tol=%.1e, itMax=%d)\n", tol, itMax)
iter = 0; errP = 2*tol

while errP > tol && iter < itMax
    global iter; global errP
    @parallel compute_Darcyflux!(qDx, qDy, qDx2, qDy2, Pf, Tf_dummy,
                                  k_eta, theta_r, _1_theta_r, _dx, _dy, 0.0)
    @parallel update_Pf!(Pf, Pold, qDx, qDy, dτ_ρ, _dt, _dx, _dy, _betaphi)

    @parallel (1:ny) bc_dirichlet_x!(Pf, P_left, P_right)
    @parallel (1:nx) bc_neumann_y!(Pf)

    iter += 1
    if iter % nout == 0
        @parallel check_res!(ResP, Pf, Pold, qDx2, qDy2, _dt, _dx, _dy, _betaphi)
        errP = norm(ResP) / sqrt(length(ResP))
        @printf("  iter = %5d  |  errP = %.3e\n", iter, errP)
    end
end

if errP <= tol
    @printf("Converged in %d iterations (errP = %.2e)\n", iter, errP)
else
    @printf("WARNING: did not converge after %d iterations (errP = %.2e)\n", iter, errP)
end

# ===========================================================================
#  COMPARISON
# ===========================================================================
p_ana = analytical_pressure(xc, yc, nx, ny, alpha_frac, b, L_ana,
                              K_m, K_f, mu_fluid, q0_ana)
p_num = Array(Pf)

# Diagonal profile
dx_diag    = sqrt(dx^2 + dy^2)
x_diag     = [dx_diag * (i-1) for i in 1:nx]
p_num_diag = [p_num[i, i] for i in 1:nx]
p_ana_diag = [p_ana[i, i] for i in 1:nx]

# ── 1. Diagonal metrics  ─────────────
#   NRMSE = RMSE / (max − min)  applied to the 1D diagonal profile
#   NMAE  = MAE  / (max − min)  applied to the 1D diagonal profile
err_diag     = p_num_diag .- p_ana_diag
p_range_diag = maximum(p_ana_diag) - minimum(p_ana_diag)
NRMSE_diag   = sqrt(mean(err_diag.^2))        / p_range_diag * 100
NMAE_diag    = mean(abs.(err_diag))            / p_range_diag * 100

# ── 2. Global 2D metrics ──────────────────────────────────────────────────────
err_2D     = p_num .- p_ana
p_range_2D = maximum(p_ana) - minimum(p_ana)
NRMSE_2D   = sqrt(mean(err_2D.^2))            / p_range_2D * 100
NMAE_2D    = mean(abs.(err_2D))               / p_range_2D * 100

# ── 3. Perturbation-only diagonal metrics  (fracture signal isolated) ─────────
xc_arr      = Array(xc)
p_lin_diag  = P_left .+ (P_right - P_left) .* xc_arr ./ lx
p_pert_ana  = p_ana_diag .- p_lin_diag
p_pert_num  = p_num_diag .- p_lin_diag
p_range_pert = maximum(p_pert_ana) - minimum(p_pert_ana)
NRMSE_pert  = sqrt(mean((p_pert_num .- p_pert_ana).^2)) / p_range_pert * 100
NMAE_pert   = mean(abs.(p_pert_num .- p_pert_ana))      / p_range_pert * 100

@printf("\n── Benchmark results ──────────────────────────────────────────────────\n")
@printf("  Metric definitions\n")
@printf("    NRMSE = RMSE / (max−min)   [Jansen/THERMAID convention]\n")
@printf("    NMAE  = MAE  / (max−min)\n")
@printf("    Range (diagonal) = %.6f MPa\n", p_range_diag/1e6)
@printf("    Range (2D field) = %.6f MPa\n", p_range_2D/1e6)
@printf("    Range (pert only)= %.6f MPa  [fracture signal w/o background]\n\n",
        p_range_pert/1e6)
@printf("  ── Diagonal (1D, as in Jansen) ──────────────────────────────────\n")
@printf("    NRMSE          : %7.4f %%\n", NRMSE_diag)
@printf("    NMAE           : %7.4f %%\n", NMAE_diag)
@printf("    RMSE           : %.4e MPa\n", sqrt(mean(err_diag.^2))/1e6)
@printf("    max|error|     : %.4e MPa\n", maximum(abs.(err_diag))/1e6)
@printf("  ── Global 2D field ──────────────────────────────────────────────\n")
@printf("    NRMSE_2D       : %7.4f %%\n", NRMSE_2D)
@printf("    NMAE_2D        : %7.4f %%\n", NMAE_2D)
@printf("    max|error|_2D  : %.4e MPa\n", maximum(abs.(err_2D))/1e6)
@printf("  ── Perturbation only (fracture signal) ──────────────────────────\n")
@printf("    NRMSE_pert     : %7.4f %%   ← accuracy on fracture perturbation\n", NRMSE_pert)
@printf("    NMAE_pert      : %7.4f %%\n", NMAE_pert)
@printf("  ── Grid ─────────────────────────────────────────────────────────\n")
@printf("    %d × %d cells,  dx = %.5f m\n", nx, ny, dx)
@printf("──────────────────────────────────────────────────────────────────────\n")

# ===========================================================================
#  PLOTS
# ===========================================================================

# Fracture tip locations along diagonal axis
x_tip1_diag = sqrt(2) * (x1_f - Array(xc)[1])
x_tip2_diag = sqrt(2) * (x2_f - Array(xc)[1])

plt1 = plot(x_diag, p_ana_diag ./ 1e6,
    label="Analytical", color=:red, linewidth=2,
    xlabel="Distance from bottom-left corner [m]",
    ylabel="Matrix Pressure [MPa]",
    title="Hydraulic benchmark",
    legend=:topleft, framestyle=:box)
scatter!(plt1, x_diag[1:6:end], p_num_diag[1:6:end] ./ 1e6,
    label="Numerical", color=:black, markersize=5)
#vline!(plt1, [x_tip1_diag, x_tip2_diag], color=:blue, linestyle=:dash, linewidth=1.5, label="Fracture tips")
annotate!(plt1, x_diag[end]*0.55, maximum(p_ana_diag)/1e6*0.8,
    "NRMSE=$(round(NRMSE_diag,digits=3))%\nNMAE =$(round(NMAE_diag, digits=3))%\n(range-norm., diagonal)")

plt2 = heatmap(Array(xc), Array(yc), p_ana' ./ 1e6,
    title="Analytical Pressure [MPa]",
    xlabel="x [m]", ylabel="y [m]",
    xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
    aspect_ratio=1, c=:RdBu, framestyle=:box)

plt3 = heatmap(Array(xc), Array(yc), p_num' ./ 1e6,
    title="Numerical Pressure [MPa]",
    xlabel="x [m]", ylabel="y [m]",
    xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
    aspect_ratio=1, c=:RdBu, framestyle=:box)

plt4 = heatmap(Array(xc), Array(yc), abs.(p_num .- p_ana)' ./ 1e6,
    title="|Error| [MPa]",
    xlabel="x [m]", ylabel="y [m]",
    xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
    aspect_ratio=1, c=:hot, framestyle=:box)

# Permeability field: verify fracture geometry visually
plt5 = heatmap(Array(xc), Array(yc), log10.(K_cpu)',
    title="log10(K) — fracture geometry check",
    xlabel="x [m]", ylabel="y [m]",
    aspect_ratio=1, c=:thermal, framestyle=:box,
    clims=(log10(K_m)-0.1, log10(K_f)+0.1))
scatter!(plt5, [x1_f, x2_f], [y1_f, y2_f],
    color=:cyan, markersize=8, markershape=:star5,
    label="Analytical tips", legend=:topright)

fullfig = plot(plt1, plt4, layout=(1,2), size=(900, 900))
display(fullfig)
