# =============================================================================
#  earthquake_detection_algo_new3.jl
#
#  Detection of numerical earthquakes and computation of their moment magnitude.
#
#  Method
#    1. A cell is "active" when its deviatoric strain rate jumps by more than
#       `threshold` in one time step: Δε̇2_dev = ε̇2_dev(t) - ε̇2_dev(t-dt).
#    2. Active cells are grouped into connected patches (4-connectivity).
#    3. A patch touching a cell of an ongoing event extends that event;
#       otherwise it starts a new event.
#    4. An event ends when none of its last active cells is still active, or
#       after `max_duration` seconds.

#    5. Rupture size L = diagonal of the bounding box of all activated cells
#       (at least one cell diagonal), circular rupture area A = π (L/2)²,
#       seismic moment M0 = G · A · ū, Mw = 2/3 · log10(M0) − 6.07.
#
# =============================================================================
using StaticArrays
using LinearAlgebra: dot, norm

function label_components(mask)
    labels = zeros(Int, size(mask))
    current_label = 0
    nx, ny = size(mask)
    visited = falses(nx, ny)
    stack = Vector{Tuple{Int, Int}}()

    for i in 1:nx, j in 1:ny
        if mask[i, j] && !visited[i, j]
            current_label += 1
            empty!(stack)
            push!(stack, (i, j))

            while !isempty(stack)
                x, y = pop!(stack)
                if x < 1 || x > nx || y < 1 || y > ny || visited[x, y] || !mask[x, y]
                    continue
                end
                visited[x, y] = true
                labels[x, y] = current_label

                push!(stack, (x-1, y))
                push!(stack, (x+1, y))
                push!(stack, (x, y-1))
                push!(stack, (x, y+1))
            end
        end
    end
    return labels
end

# Cell-centre value of a node field (size nx+1 × ny+1) for cell I (size nx × ny)
@inline node2cell(A, I) = 0.25 * (A[I[1], I[2]] + A[I[1]+1, I[2]] + A[I[1], I[2]+1] + A[I[1]+1, I[2]+1])

# -----------------------------------------------------------------------------
#  detect_seismic_events_fast!  —  call once per time step
#
#  Inputs
#    ε2_dev, ε2_dev_prev : deviatoric strain rate at t and t-dt          [1/s]
#    Vx, Vy, Ux, Uy      : nodal velocities [m/s] and displacements [m]
#    angle_2d            : CPU array, fracture angle from vertical       [°]
#    material            : CPU array, 1 = matrix, >1 = fracture
#    time, dt            : current time and time step                     [s]
#    dx, dy              : cell size                                      [m]
#    G                   : shear modulus                                  [Pa]
#    xc, yc              : cell-centre coordinates                        [m]
#  State (persist between calls, created empty in the main script)
#    event_state, event_catalog, event_U0_store, cell_to_event, next_event_id_ref
#  Outputs (appended when an event ends)
#    Mw_list      : moment magnitude
#    Mw_time      : event start time                                      [s]
#    Mw_location  : (x, y) of the event (running mean of patch centroids) [m]
#    L_patch      : rupture size L                                         [m]
#  Optional keyword outputs
#    slip_list     : mean slip ū of each event                             [m]
#    Mw_list_vnorm : magnitude with the path-length slip Σ dt·|v| (comparison only)
#  Keyword parameters
#    threshold     : activation threshold on Δε̇2_dev                     [1/s]
#    max_duration  : maximum event duration                               [s]
# -----------------------------------------------------------------------------
function detect_seismic_events_fast!(
    ε2_dev,
    ε2_dev_prev,
    Vx,
    Vy,
    Ux,
    Uy,
    angle_2d::AbstractMatrix,
    material::AbstractMatrix,
    event_state::Dict{Int,Bool},
    event_catalog::Dict{Int,Dict},
    event_U0_store::Dict{Int, Dict{CartesianIndex{2}, SVector{2,Float64}}},
    cell_to_event::Dict{CartesianIndex{2}, Int},
    time::Float64,
    dt::Float64,
    dx::Float64,
    dy::Float64,
    G::Float64,
    Mw_list,
    Mw_time,
    Mw_location::Vector{Tuple{Float64,Float64}},
    L_patch,
    next_event_id_ref::Base.RefValue{Int},
    xc::LinRange{Float64, Int64},
    yc::LinRange{Float64, Int64};
    threshold::Float64    = 1e-11,
    max_duration::Float64 = 20.0,
    slip_list             = nothing,
    Mw_list_vnorm         = nothing
)
    # ---- 1. active cells ----------------------------------------------------
    Δε2 = ε2_dev .- ε2_dev_prev
    active_mask = Array(Δε2 .> threshold)

    # ---- 2. connected patches -------------------------------------------------
    labels   = label_components(active_mask)
    n_labels = maximum(labels; init=0)

    # nothing active and no open event → skip (avoids GPU → CPU copies)
    if n_labels == 0 && !any(values(event_state))
        return
    end

    # one GPU → CPU copy per call instead of scalar indexing on the GPU
    Ux_h, Uy_h = Array(Ux), Array(Uy)
    Vx_h, Vy_h = Array(Vx), Array(Vy)

    label_regions = [CartesianIndex{2}[] for _ in 1:n_labels]
    for I in CartesianIndices(labels)
        l = labels[I]
        l > 0 && push!(label_regions[l], I)
    end

    # ---- 3. link each patch to an ongoing event or start a new one ------------
    for label_id in 1:n_labels
        region = label_regions[label_id]
        isempty(region) && continue

        x̄ = mean(I -> xc[I[1]], region)
        ȳ = mean(I -> yc[I[2]], region)

        linked_event = 0
        for I in region
            eid = get(cell_to_event, I, 0)
            if eid != 0 && get(event_state, eid, false)
                linked_event = eid
                break
            end
        end

        if linked_event == 0
            # new event
            eid = next_event_id_ref[]
            next_event_id_ref[] += 1

            event_state[eid] = true
            event_catalog[eid] = Dict(
                :start_time       => time,
                :active_locations => Set(region),
                :count            => length(region),
                :x_mean           => x̄,
                :y_mean           => ȳ,
                :x_min            => minimum(I -> xc[I[1]], region),
                :x_max            => maximum(I -> xc[I[1]], region),
                :y_min            => minimum(I -> yc[I[2]], region),
                :y_max            => maximum(I -> yc[I[2]], region),
                :u_path           => 0.0,     # Σ dt·|v| (comparison only)
            )
            event_U0_store[eid] = Dict{CartesianIndex{2}, SVector{2,Float64}}()

            for I in region
                cell_to_event[I] = eid
            end
        else
            # extend the ongoing event
            eid  = linked_event
            info = event_catalog[eid]
            info[:count] += length(region)
            info[:x_mean] = (info[:x_mean] + x̄) / 2
            info[:y_mean] = (info[:y_mean] + ȳ) / 2

            info[:x_min] = min(info[:x_min], minimum(I -> xc[I[1]], region))
            info[:x_max] = max(info[:x_max], maximum(I -> xc[I[1]], region))
            info[:y_min] = min(info[:y_min], minimum(I -> yc[I[2]], region))
            info[:y_max] = max(info[:y_max], maximum(I -> yc[I[2]], region))

            for I in info[:active_locations]
                delete!(cell_to_event, I)
            end
            info[:active_locations] = Set(region)
            for I in region
                cell_to_event[I] = eid
            end
        end

        U0_store = event_U0_store[eid]
        info = event_catalog[eid]
        for I in region
            vx = node2cell(Vx_h, I); vy = node2cell(Vy_h, I)
            get!(U0_store, I) do
                SVector(node2cell(Ux_h, I) - dt * vx, node2cell(Uy_h, I) - dt * vy)
            end
            info[:u_path] += dt * sqrt(vx^2 + vy^2)
        end
    end

    # ---- 4. close finished events and compute their magnitude -----------------
    to_finalize = Int[]

    for (eid, info) in event_catalog
        event_state[eid] || continue

        elapsed      = time - info[:start_time]
        timed_out    = elapsed >= max_duration
        still_active = !timed_out && any(I -> active_mask[I], info[:active_locations])

        if !still_active
            # mean net slip, projected on the fracture trace
            U0_store = event_U0_store[eid]
            slip_sum = 0.0
            for (I, U0) in U0_store
                ΔU = SVector(node2cell(Ux_h, I), node2cell(Uy_h, I)) - U0
                if material[I] > 1
                    α = angle_2d[I] * π / 180
                    t = SVector(sin(α), -cos(α))     # down-dip fracture tangent
                    slip_sum += abs(dot(ΔU, t))
                else
                    slip_sum += norm(ΔU)             # matrix: no plane defined
                end
            end
            n_cells = length(U0_store)
            u_avg   = n_cells > 0 ? slip_sum / n_cells : 0.0

            # rupture size and area
            L = sqrt((info[:x_max] - info[:x_min])^2 + (info[:y_max] - info[:y_min])^2)
            L = max(L, sqrt(dx^2 + dy^2))
            A = pi * (L / 2)^2

            # seismic moment and moment magnitude
            M0 = G * A * u_avg
            Mw = (2/3) * log10(max(M0, 1e-30)) - 6.07

            push!(Mw_list,     Mw)
            push!(Mw_time,     info[:start_time])
            push!(Mw_location, (info[:x_mean], info[:y_mean]))
            push!(L_patch,     L)
            slip_list === nothing || push!(slip_list, u_avg)
            if Mw_list_vnorm !== nothing
                u_old = n_cells > 0 ? info[:u_path] / n_cells : 0.0
                push!(Mw_list_vnorm, (2/3) * log10(max(G * A * u_old, 1e-30)) - 6.07)
            end

            for I in info[:active_locations]
                delete!(cell_to_event, I)
            end
            event_state[eid] = false
            push!(to_finalize, eid)
        end
    end

    for eid in to_finalize
        delete!(event_catalog,  eid)
        delete!(event_U0_store, eid)
        delete!(event_state,    eid)
    end

    return
end
