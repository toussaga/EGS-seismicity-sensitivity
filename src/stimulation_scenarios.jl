# =============================================================================
#  stimulation_scenarios.jl
#
#  Injection protocols. Each function imposes the fluid overpressure Pf [Pa]
#  (above hydrostatic) and the fluid temperature Tf [K] at the injection cells
#  for the current time [s], and returns (Pf, Tf).
#
#    stimulate_smoothstep_pressure!  monotonic step increase (main scenarios)
#    stimulate_ccs_pressure!         cyclic soft stimulation (Pohang-type)
#    compute_total_flow              injected flow rate through the injection cells
# =============================================================================

# Step-wise pressure increase with smooth (cosine) transitions:
#   +ΔP every step_duration, each step reached over ramp_duration.
#   Default: +5 MPa per day, 4 h ramps, injection at 60 °C.
function stimulate_smoothstep_pressure!(Pf, Tf, inj_points, time;
    P_start       = 100,            # initial overpressure     [Pa]
    ΔP            = 5e6,            # pressure increase/step   [Pa]
    step_duration = 3600*24,        # step length              [s]
    ramp_duration = 3600*4,         # transition length        [s]
    T_inj         = 60.0 + 273.15)  # injection temperature    [K]

    step_number = floor(Int, time / step_duration)
    t_in_step   = time - step_number * step_duration

    # cosine smoothing (C1 continuous)
    f = t_in_step < ramp_duration ? 0.5 * (1 - cos(pi * t_in_step / ramp_duration)) : 1.0

    P_now = P_start + step_number * ΔP + f * ΔP

    Pf = assign_P0(Pf, inj_points, P_now)
    Tf = assign_P0(Tf, inj_points, T_inj)
    return Pf, Tf
end

# Net Darcy flux leaving the injection cells, summed over their four faces.
# Each face contributes half of flux × face length (factor 0.5).
# Result in m²/s per metre of out-of-plane thickness (2D model).
function compute_total_flow(qDx2, qDy2, inj_points, dx, dy)
    Qinj = 0.0
    @allowscalar for (i, j) in inj_points
        # left face, between (i-1, j) and (i, j)
        if i > 1 && j > 1 && j <= size(qDx2,2)
            Qinj -= qDx2[i-1, j-1] * dy * 0.5
        end
        # right face, between (i, j) and (i+1, j)
        if i <= size(qDx2,1) && j > 1 && j <= size(qDx2,2)
            Qinj += qDx2[i, j-1] * dy * 0.5
        end
        # bottom face, between (i, j-1) and (i, j)
        if j > 1 && i > 1 && i <= size(qDy2,1)
            Qinj -= qDy2[i-1, j-1] * dx * 0.5
        end
        # top face, between (i, j) and (i, j+1)
        if j <= size(qDy2,2) && i > 1 && i <= size(qDy2,1)
            Qinj += qDy2[i-1, j] * dx * 0.5
        end
    end
    return Qinj
end
