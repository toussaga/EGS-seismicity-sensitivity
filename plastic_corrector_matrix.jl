# =============================================================================
#  plastic_corrector_matrix!
#
#  Plastic return mapping for ROCK MATRIX cells (material == 1).
#
#  Drucker-Prager type criterion with tension cut-off:
#    - shear   : Fs = τ2 - c·cos(φ) + P·sin(φ)
#    - tensile : Ft = P - T0
#  where τ2 is the second invariant of the deviatoric stress and P the mean
#  stress (compression negative). The active mode (largest of Fs, Ft) is
#  corrected iteratively with a non-associated flow rule (dilation angle ψ).
#
#  Inputs
#    Sxx_check, Syy_check, Txy_check : trial deviatoric stresses        [Pa]
#    P                               : trial mean stress (overwritten)  [Pa]
#    material                        : material id (1 = matrix, >1 = fracture)
#    Cohesion, phi, psi              : cohesion [Pa], friction and dilation angles [°]
#    G, K                            : shear and bulk moduli            [Pa]
#    tensile                         : tensile strength                 [Pa]
#    dt                              : time step                        [s]
#
#  Outputs (written for matrix cells only)
#    Sxx_dev, Syy_dev, Txy_dev, P : corrected stresses                  [Pa]
#    Tau2                         : second invariant of the deviator    [Pa]
#    ε2_pl                        : plastic strain RATE (shear + tensile) [1/s]
#    F_check                      : yield function after correction (≈ 0 if converged)
#    F_fail                       : yield function BEFORE correction, max(Fs, Ft)
#    F_type                       : 0 = elastic, 1 = shear failure, 2 = tensile failure
#
#  Numerical notes
#    - Each iteration applies only the increment δλ (incremental update).
#    - tolF = 1e-3 Pa is above round-off error for stresses of ~1e7 Pa.
# =============================================================================
@parallel_indices (ix,iy) function plastic_corrector_matrix!(
    Sxx_check, Syy_check, Txy_check, P,               # trial stresses
    Sxx_dev, Syy_dev, Txy_dev,                        # corrected stresses
    Tau2, F_check, ε2_pl, F_fail, F_type, material,   # diagnostics
    Cohesion, phi, psi, G, K, tensile, dt             # scalars
)
    tolF     = 1e-3          # convergence tolerance on the yield function [Pa]
    max_iter = 20

    if material[ix,iy] == 1
        sφ = sin(phi * π/180)
        cφ = cos(phi * π/180)
        sψ = sin(psi * π/180)

        τx  = Sxx_check[ix,iy]
        τy  = Syy_check[ix,iy]
        τxy = Txy_check[ix,iy]
        Pij = P[ix,iy]

        # second invariant of the deviator (includes the out-of-plane component -τx-τy)
        τ2 = sqrt(0.5*(τx^2 + τy^2 + (-τx - τy)^2) + τxy^2)

        # yield functions
        Fs = τ2 - Cohesion * cφ + Pij * sφ
        Ft = Pij - tensile
        F  = max(Fs, Ft)

        ε2_p_shear = 0.0
        ε2_p_tens  = 0.0
        Δλ     = 0.0         # accumulated plastic multiplier (strain increment)
        F_test = F
        type   = 0.0
        it     = 0

        if F > 0
            if Fs >= Ft
                # ---------- shear failure ----------
                type = 1.0
                den  = G + K * sφ * sψ
                τx_tr, τy_tr, τxy_tr, τ2_tr = τx, τy, τxy, τ2   # trial state (flow direction)
                while F_test > tolF && it < max_iter
                    δλ  = F_test / den
                    Δλ += δλ

                    # radial scaling of the deviator (0 = return to the apex)
                    scale = max(1.0 - G * δλ / τ2, 0.0)
                    τx  *= scale
                    τy  *= scale
                    τxy *= scale

                    # dilatancy: change of mean stress
                    Pij -= K * δλ * sψ

                    τ2     = sqrt(0.5*(τx^2 + τy^2 + (-τx - τy)^2) + τxy^2)
                    F_test = τ2 - Cohesion * cφ + Pij * sφ
                    it += 1
                end
                # plastic strain increment along the trial direction τ/τ2
                # (same as the final direction because of radial scaling, and
                # still defined if the stress returns to the apex)
                coeff      = Δλ / τ2_tr
                εxx_p      = coeff * τx_tr / 2
                εyy_p      = coeff * τy_tr / 2
                εxy_p      = coeff * τxy_tr
                ε2_p_shear = sqrt(0.5*(εxx_p^2 + εyy_p^2 + (-εxx_p-εyy_p)^2) + εxy_p^2)
            else
                # ---------- tensile (opening) failure ----------
                type = 2.0
                while F_test > tolF && it < max_iter
                    δλ  = F_test / K
                    Δλ += δλ
                    Pij -= K * δλ
                    F_test = Pij - tensile
                    it += 1
                end
                ε2_p_tens = sqrt(0.5 * Δλ^2)
            end
        end

        Sxx_dev[ix,iy] = τx
        Syy_dev[ix,iy] = τy
        Txy_dev[ix,iy] = τxy
        P[ix,iy]       = Pij
        ε2_pl[ix,iy]   = (ε2_p_shear + ε2_p_tens) / dt   # plastic strain rate [1/s]
        Tau2[ix,iy]    = τ2
        F_check[ix,iy] = F_test
        F_fail[ix,iy]  = F
        F_type[ix,iy]  = type
    end
    return
end
