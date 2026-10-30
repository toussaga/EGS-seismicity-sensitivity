# =============================================================================
#  domain_properties.jl
#
#  Fracture geometry, material map, material properties and injection points.
#
#  Grid conventions (same as the main script):
#    xc[i] : cell-centre x coordinate [m]
#    yc[j] : cell-centre y coordinate [m], NEGATIVE below the surface
#  Material map: 1 = rock matrix, k + 1 = cells belonging to fracture k.
# =============================================================================

# One fracture in the 2D section, defined as a segment from (x0, y0) (upper
# end) to (xend, yend) (lower end).
mutable struct Fault
    x0::Float64          # x of upper end                       [m]
    y0::Float64          # y of upper end (negative)            [m]
    yend::Float64        # y of lower end (negative)            [m]
    thickness::Int64     # max. distance of a cell to the trace [m]
    dip::Float64         # dip                                  [°]
    strike::Float64      # strike                               [°]
    xend::Float64        # x of lower end                       [m]
    app_dip::Float64     # angle in the section plane, from vertical [°]
                         # (copied to angle_2d by assign_geometry_2d!)
end

# Assign hydraulic properties to every cell from its material id.
function assign_properties!(material, properties, k0, σ_star, α)
    for k in 1:size(properties, 1)
        mask = material .== k
        k0[mask]     .= properties[k, 1]
        σ_star[mask] .= properties[k, 2]
        α[mask]      .= properties[k, 3]
    end
end

# Vertical injection interval: all cells within xtol of x_target whose DEPTH
# lies between y_min and y_max (depths are positive, i.e. -y_max ≤ yc ≤ -y_min).
# Returns a vector of (i, j) cell indices.
function get_injection_points(xc, yc, nx, ny, x_target, y_min, y_max, xtol)
    points = []
    for i in 1:nx
        for j in 1:ny
            if abs(xc[i] - x_target) <= xtol && -y_max <= yc[j] <= -y_min
                push!(points, (i, j))
            end
        end
    end
    return points
end

# Horizontal injection interval: all cells within ytol of depth y_target
# (positive) and with x_min ≤ xc ≤ x_max.
function get_injection_points_horizontal(xc, yc, nx, ny, y_target, x_min, x_max, ytol)
    points = []
    for i in 1:nx
        for j in 1:ny
            if abs(yc[j] + y_target) <= ytol && x_min <= xc[i] <= x_max
                push!(points, (i, j))
            end
        end
    end
    return points
end

# Set A[i, j] = value at every (i, j) in points
function assign_P0(A, points, value)
    @allowscalar for (i, j) in points
        A[i, j] = value
    end
    return A
end

# Material map from the fracture list: a cell belongs to fracture k if its
# distance to the fracture line is ≤ thickness and it lies between the two ends in y. 
function assign_material(fractures, xc, yc, nx, ny)
    material      = ones(Int, nx, ny)
    dist_to_fault = zeros(Float64, nx, ny)

    for k in 1:length(fractures)
        fault = fractures[k]
        for i in 1:nx
            for j in 1:ny
                # distance from the cell centre to the (infinite) fracture line
                dist_to_fault[i,j] = abs((fault.yend - fault.y0) * xc[i] - (fault.xend - fault.x0) * yc[j] + fault.xend * fault.y0 - fault.yend * fault.x0) /
                                     sqrt((fault.yend - fault.y0)^2 + (fault.xend - fault.x0)^2)

                if dist_to_fault[i,j] <= fault.thickness && fault.yend <= yc[j] <= fault.y0
                    material[i, j] = 1 + k
                end
            end
        end
    end
    return material
end
