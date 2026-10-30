# =============================================================================
#  plastic_corrector_faults_2D!
#
#  Plastic return mapping for FRACTURE cells (material > 1).
#
#  The trial stress is resolved on the fracture plane, whose orientation is
#  given by angle_2d (angle from vertical, in degrees). Two failure modes are
#  checked on that plane:
#    - shear   : Coulomb criterion  Fs = |τ| + σn·tan(φ_f) - c_f   (σn < 0)
#                                   Fs = |τ| - c_f                 (σn ≥ 0)
#    - tensile : Ft = σn - T0
#  The active mode (largest of Fs, Ft) is corrected iteratively with a
#  non-associated flow rule (dilation angle ψ_f), then the corrected stress is
#  rotated back to the x-y frame and split into mean stress and deviator.
#
#  Conventions: compression negative, 2D mean stress P = (σxx + σyy)/2.
#
#  Inputs
#    Sxx_check, Syy_check, Txy_check : trial deviatoric stresses        [Pa]
#    P                               : trial mean stress (overwritten)  [Pa]
#    G, K                            : shear and bulk moduli            [Pa]
#    material                        : material id (1 = matrix, >1 = fracture)
#    angle_2d                        : orientation from vertical        [°]
#    coh_f, phi_f, psi_f             : cohesion [Pa], friction and dilation angles [°]
#    tensile                         : tensile strength                 [Pa]
#    dt                              : time step                        [s]
#
#  Outputs (written for fracture cells only)
#    Sxx_dev, Syy_dev, Txy_dev, P : corrected stresses                  [Pa]
#    Tau2                         : second invariant of the deviator    [Pa]
#    ε2_pl                        : plastic shear strain RATE           [1/s]
#                                   (increment / dt, as in the matrix corrector)
#    F_check                      : yield function after correction (≈ 0 if converged)
#    F_fail                       : shear yield function BEFORE correction (> 0 = failure)
#    F_type                       : 0 = elastic, 1 = shear failure, 2 = tensile failure
#
#  Numerical notes
#    - Each iteration applies only the increment δλ (incremental update).
#    - tolF = 1e-3 Pa is above round-off error for stresses of ~1e7 Pa.
# =============================================================================
@parallel_indices (ix,iy) function plastic_corrector_faults_2D!(
    Sxx_check, Syy_check, Txy_check, P,
    Sxx_dev, Syy_dev, Txy_dev,
    G, K, Tau2,
    F_check, ε2_pl, F_fail, F_type,
    material, angle_2d,
    coh_f, phi_f, psi_f, tensile, dt
)
    tolF     = 1e-3          # convergence tolerance on the yield function [Pa]
    max_iter = 15
    deg2rad  = π/180
    α1 = K + 4 * G / 3       # P-wave (constrained) modulus
    α2 = K - 2 * G / 3
    tφ = tan(phi_f * deg2rad)
    tψ = tan(psi_f * deg2rad)

    τx  = Sxx_check[ix,iy]
    τy  = Syy_check[ix,iy]
    τxy = Txy_check[ix,iy]
    Pij = P[ix,iy]

    if material[ix,iy] > 1
        ε2_p_shear = 0.0
        ε2_p_tens  = 0.0     # tensile/dilatant part (computed, not stored)
        Δλ   = 0.0           # accumulated plastic multiplier
        type = 0.0
        Fs, Ft = 0.0, 0.0

        # --- fracture frame: unit normal n and tangent t ---
        α   = angle_2d[ix,iy] * deg2rad
        nx_ =  cos(α)
        ny_ = -sin(α)
        tx_ =  sin(α)
        ty_ =  cos(α)

        # --- total trial stress ---
        σxx = τx + Pij
        σyy = τy + Pij
        σxy = τxy

        # --- components in the fracture frame ---
        tx_tr = σxx * nx_ + σxy * ny_            # traction t_i = σ_ij n_j
        ty_tr = σxy * nx_ + σyy * ny_
        σnn      = tx_tr * nx_ + ty_tr * ny_     # normal stress
        τshear_s = tx_tr * tx_ + ty_tr * ty_     # signed shear stress
        τshear   = abs(τshear_s)
        τ_sign   = τshear_s >= 0.0 ? 1.0 : -1.0
        σtt      = σxx * tx_ * tx_ + 2.0 * σxy * tx_ * ty_ + σyy * ty_ * ty_   # stress parallel to the fracture

        # --- yield functions ---
        if σnn >= 0.0
            Fs = τshear - coh_f                  # no frictional strength in tension
        else
            Fs = τshear + σnn * tφ - coh_f       # Coulomb
        end
        Ft = σnn - tensile
        F  = max(Fs, Ft)
        F_test = F
        it = 0

        if F > 0
            if Fs >= Ft
                # ---------- shear failure ----------
                type = 1.0
                den  = 2 * G + α1 * tφ * tψ
                while F_test > tolF && it < max_iter
                    δλ  = F_test / den
                    Δλ += δλ

                    τshear = max(τshear - 2 * G * δλ, 0.0)   # shear stress drop

                    div_p      = δλ * tψ                     # dilatant volumetric strain
                    ε2_p_tens += sqrt(0.5 * div_p^2)
                    σnn -= α1 * div_p
                    σtt -= α2 * div_p

                    if σnn >= 0.0
                        F_test = τshear - coh_f
                    else
                        F_test = τshear + σnn * tφ - coh_f
                    end
                    it += 1
                end
                ε2_p_shear = Δλ
            else
                # ---------- tensile (opening) failure ----------
                type = 2.0
                while F_test > tolF && it < max_iter
                    δλ  = F_test / α1
                    Δλ += δλ

                    div_p      = δλ
                    ε2_p_tens += sqrt(0.5 * div_p^2)
                    σnn -= α1 * div_p
                    σtt -= α2 * div_p

                    F_test = σnn - tensile
                    it += 1
                end
            end
        end

        # --- rotate corrected stress back to the x-y frame ---
        τ_signed = τshear * τ_sign
        σxx = σnn * nx_ * nx_ + σtt * tx_ * tx_ + 2 * τ_signed * nx_ * tx_
        σyy = σnn * ny_ * ny_ + σtt * ty_ * ty_ + 2 * τ_signed * ny_ * ty_
        σxy = σnn * nx_ * ny_ + σtt * tx_ * ty_ +     τ_signed * (nx_ * ty_ + ny_ * tx_)

        # --- mean stress and deviator (2D) ---
        Pij = 0.5 * (σxx + σyy)
        τx  = σxx - Pij
        τy  = σyy - Pij
        τxy = σxy
        τ2  = sqrt(0.5*(τx^2 + τy^2) + τxy^2)

        Sxx_dev[ix,iy] = τx
        Syy_dev[ix,iy] = τy
        Txy_dev[ix,iy] = τxy
        P[ix,iy]       = Pij
        ε2_pl[ix,iy]   = (ε2_p_shear + ε2_p_tens) / dt     # plastic shear strain rate [1/s]
        Tau2[ix,iy]    = τ2
        F_check[ix,iy] = F_test
        F_fail[ix,iy]  = F
        F_type[ix,iy]  = type
    end
    return
end
