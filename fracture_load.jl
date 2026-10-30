# =============================================================================
#  Reading the fracture network and assigning fracture orientations to the grid.
#
#  Fracture network CSV format (one row per fracture):
#    x1, y1, x2, y2 : end points of the fracture trace in the section   [m]
#                     (y negative below the surface, same frame as yc)
#    dip, strike    : 3D orientation [°], kept for reference only
#    family         : family id (1, 2, 3 …), used to assign FAMILY_PROPS
#
#  Angle convention (used everywhere in the code): the 2D angle α is measured
#  from the vertical, α = atan(Δx / |Δy|) from the upper to the lower end point,
#    positive = dips towards +x, negative = dips towards -x.
#  Fracture tangent (pointing down-dip): t = ( sin α, -cos α)
#  Fracture normal:                      n = ( cos α,  sin α)
# =============================================================================
using CSV, DataFrames

# Read the fracture network.
# Returns the vector of Fault and the family id of each fracture.
function load_fractures_2d(filename::String, thickness::Float64)
    df = CSV.read(filename, DataFrame)
    Faults = Vector{Fault}(undef, nrow(df))
    for (i, row) in enumerate(eachrow(df))
        # convention: (x0, y0) is the upper (shallower) end point
        if row.y1 > row.y2
            x0, y0, xend, yend = row.x1, row.y1, row.x2, row.y2
        else
            x0, y0, xend, yend = row.x2, row.y2, row.x1, row.y1
        end

        # 2D angle from vertical [°]
        dy = abs(y0 - yend)
        dx = xend - x0
        angle = dy > 0 ? atand(dx / dy) : 0.0

        Faults[i] = Fault(x0, y0, yend, thickness,
                          row.dip, row.strike,     # reference only
                          xend, angle)
    end
    return Faults, df.family
end

# Copy the 2D angle of each fracture to the grid cells it covers.
function assign_geometry_2d!(material, Faults, angle_2d)
    nx, ny = size(angle_2d)
    for (k, fault) in enumerate(Faults)
        for i in 1:nx, j in 1:ny
            if material[i,j] == k + 1
                angle_2d[i, j] = fault.app_dip     # angle computed in load_fractures_2d
            end
        end
    end
    return angle_2d
end
