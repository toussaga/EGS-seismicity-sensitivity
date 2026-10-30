# =============================================================================
#  gen_frac_networks.jl
#
#  Generates the 2D fracture networks used in the sensitivity analysis.
#
#  One conjugate set of two fracture families, symmetric about the vertical:
#    family 1 : angle +HALF_ANGLE from vertical (dips towards +x)
#    family 2 : angle -HALF_ANGLE from vertical (dips towards -x)
#  Each fracture has
#    - a random centre in [XMIN, XMAX] × [YMIN, YMAX],
#    - an angle drawn from Normal(±HALF_ANGLE, ANGLE_SIGMA),
#    - a length drawn from a truncated power law p(l) ∝ l^(-ALPHA_PL) on
#      [LMIN, LMAX],
#    - a centre at least DMIN from the centres of fractures already placed.
#  All lengths are then multiplied by a common factor so that the total
#  fracture length equals L_TARGET. Connected clusters (intersecting
#  fractures) are identified and written to the CSV for information.
#
#  Two networks are written, one per stress regime:
#    NF : the generated network
#    TF : the NF network rotated by 90° about the centre of the domain
#         (sub-vertical fractures become sub-horizontal)
#
#  Conventions
#    - y negative below the surface [m]
#    - angle measured from the vertical, positive = dips towards +x
#    - in the CSV, (x1, y1) is the upper (shallower) end point
#
#  Output CSV columns: id, x1, y1, x2, y2, dip, strike, family, cluster
#
# =============================================================================
using Random, CSV, DataFrames

# =============================================================================
#  PARAMETERS
# =============================================================================

# --- Output ---
const NETWORK_TAG = "short06"     # appended to the file names: fractures_<regime>_<tag>.csv
const OUTPUT_DIR  = "."           # folder where the CSV files are written

# --- Region where fractures are placed (inside the model domain) ---
const XMIN, XMAX =  150.0,  1850.0   # [m]
const YMIN, YMAX = -5350.0, -3650.0  # [m]

# --- Orientation ---
const HALF_ANGLE  = 30.0   # mean angle of each family from vertical [°] (30° → 60° dip)
const ANGLE_SIGMA = 10.0   # standard deviation of the angle [°]

# --- Number, lengths and spacing ---
const N_FRACS_PER_FAMILY = 90
const LMIN     = 400.0      # min. length BEFORE rescaling [m]
const LMAX     = 1500.0     # max. length BEFORE rescaling [m]
const ALPHA_PL = 1.6       # power-law exponent of the length distribution
const DMIN     = 150.0     # min. distance between fracture centres [m]
const L_TARGET = 35500.0   # total fracture length after rescaling [m]

const SEED = 80

# --- dip / strike metadata written to the CSV ---
const NF_DIP, NF_STRIKE = 60.0, 144.0   # [°]
const SS_DIP, SS_STRIKE = 80.0, 144.0   # [°]
const TF_DIP, TF_STRIKE = 30.0, 144.0   # [°]

# =============================================================================
#  HELPERS
# =============================================================================

# Sample from a truncated power law p(l) ∝ l^(-alpha) on [lmin, lmax]
function sample_powerlaw(lmin, lmax, alpha)
    a = 1.0 - alpha
    u = rand()
    return ((lmax^a - lmin^a) * u + lmin^a)^(1/a)
end

# True if segments (1-2) and (3-4) intersect
function segments_intersect(x1,y1,x2,y2, x3,y3,x4,y4)
    ccw(ax,ay,bx,by,cx,cy) = (cy-ay)*(bx-ax) > (by-ay)*(cx-ax)
    return (ccw(x1,y1,x3,y3,x4,y4) != ccw(x2,y2,x3,y3,x4,y4)) &&
           (ccw(x1,y1,x2,y2,x3,y3) != ccw(x1,y1,x2,y2,x4,y4))
end

# Union-find, used to group intersecting fractures into clusters
mutable struct UF; parent::Vector{Int}; rank::Vector{Int}; end
make_uf(n) = UF(collect(1:n), zeros(Int,n))
function uf_find(uf, x)
    uf.parent[x] != x && (uf.parent[x] = uf_find(uf, uf.parent[x]))
    return uf.parent[x]
end
function uf_union!(uf, a, b)
    ra, rb = uf_find(uf,a), uf_find(uf,b)
    ra == rb && return
    if     uf.rank[ra] < uf.rank[rb]; uf.parent[ra] = rb
    elseif uf.rank[ra] > uf.rank[rb]; uf.parent[rb] = ra
    else   uf.parent[rb] = ra; uf.rank[ra] += 1
    end
end

# Angle from vertical [°] of a segment, upper point (x1, y1) first
angle_from_vertical(x1, y1, x2, y2) = abs(y1 - y2) > 0 ? atand((x2 - x1) / abs(y1 - y2)) : 0.0

# =============================================================================
#  NETWORK GENERATION
# =============================================================================

struct Frac2D
    id      :: Int
    x1      :: Float64   # upper end point (y1 > y2)
    y1      :: Float64
    x2      :: Float64   # lower end point
    y2      :: Float64
    angle   :: Float64   # angle from vertical [°] (not exported)
    dip     :: Float64   # metadata
    strike  :: Float64   # metadata
    family  :: Int
    cluster :: Int
end

# Draw n fractures of one family with mean angle angle_mean (from vertical, [°]).
# Candidates are rejected if they leave the placement region or if their centre is closer than dmin to an existing centre.
# Returns tuples (xm, ym, angle, length, x_up, y_up, x_down, y_down, family).
function gen_family(n::Int, family_id::Int, angle_mean::Float64;
                    sigma=ANGLE_SIGMA, lmin=LMIN, lmax=LMAX,
                    alpha=ALPHA_PL, dmin=DMIN, max_tries=20_000)
    fracs = []
    tries = 0
    while length(fracs) < n && tries < max_tries
        tries += 1
        xm = XMIN + rand()*(XMAX - XMIN)
        ym = YMIN + rand()*(YMAX - YMIN)

        ang = angle_mean + randn()*sigma
        l   = sample_powerlaw(lmin, lmax, alpha)

        θ  = deg2rad(ang)
        dx = (l/2) * sin(θ)          # horizontal half-extent
        dy = (l/2) * cos(θ)          # vertical half-extent
        xu, yu = xm - dx, ym + dy    # upper end point
        xd, yd = xm + dx, ym - dy    # lower end point

        (xu<XMIN || xu>XMAX || xd<XMIN || xd>XMAX ||
         yu<YMIN || yu>YMAX || yd<YMIN || yd>YMAX) && continue
        any(sqrt((xm-f[1])^2+(ym-f[2])^2) < dmin for f in fracs) && continue

        push!(fracs, (xm, ym, ang, l, xu, yu, xd, yd, family_id))
    end
    length(fracs) < n &&
        @warn "Family $family_id: only $(length(fracs))/$n fractures placed after $max_tries tries"
    return fracs
end

# Rescale all lengths by `scale` about their centres and attach dip/strike metadata.
function build_network(raw, scale, dip, strike)
    fracs = Frac2D[]
    for (id, (xm, ym, ang, l, xu, yu, xd, yd, fam)) in enumerate(raw)
        l2 = l * scale
        θ  = deg2rad(ang)
        dx = (l2/2) * sin(θ)
        dy = (l2/2) * cos(θ)
        push!(fracs, Frac2D(id, xm - dx, ym + dy, xm + dx, ym - dy, ang, dip, strike, fam, -1))
    end
    return fracs
end

# Label connected clusters of intersecting fractures
function assign_clusters(fracs::Vector{Frac2D})
    n  = length(fracs)
    uf = make_uf(n)
    for i in 1:n-1, j in i+1:n
        segments_intersect(fracs[i].x1, fracs[i].y1, fracs[i].x2, fracs[i].y2,
                           fracs[j].x1, fracs[j].y1, fracs[j].x2, fracs[j].y2) &&
            uf_union!(uf, i, j)
    end
    cmap = Dict{Int,Int}(); cid = 0
    return [begin
                root = uf_find(uf, i)
                !haskey(cmap, root) && (cid += 1; cmap[root] = cid)
                Frac2D(f.id, f.x1, f.y1, f.x2, f.y2,
                       f.angle, f.dip, f.strike, f.family, cmap[root])
            end for (i, f) in enumerate(fracs)]
end

# Rigid 90° counter-clockwise rotation about the centre of the placement region
# (used to build the TF network from the NF network). Clusters are preserved.
function rotate_90(fracs::Vector{Frac2D}, dip, strike)
    cx = (XMIN + XMAX) / 2.0
    cy = (YMIN + YMAX) / 2.0
    rot(x, y) = (cx - (y - cy), cy + (x - cx))
    return [begin
                x1r, y1r = rot(f.x1, f.y1)
                x2r, y2r = rot(f.x2, f.y2)
                # keep the upper end point first
                if y1r >= y2r
                    x1f, y1f, x2f, y2f = x1r, y1r, x2r, y2r
                else
                    x1f, y1f, x2f, y2f = x2r, y2r, x1r, y1r
                end
                Frac2D(f.id, x1f, y1f, x2f, y2f,
                       angle_from_vertical(x1f, y1f, x2f, y2f),
                       dip, strike, f.family, f.cluster)
            end for f in fracs]
end

# =============================================================================
#  EXPORT
# =============================================================================

function export_csv(fracs::Vector{Frac2D}, filename::String)
    df = DataFrame(
        id      = [f.id      for f in fracs],
        x1      = [f.x1      for f in fracs],
        y1      = [f.y1      for f in fracs],
        x2      = [f.x2      for f in fracs],
        y2      = [f.y2      for f in fracs],
        dip     = [f.dip     for f in fracs],
        strike  = [f.strike  for f in fracs],
        family  = [f.family  for f in fracs],
        cluster = [f.cluster for f in fracs],
    )
    CSV.write(filename, df)
    n_cl = length(unique(f.cluster for f in fracs))
    Ltot = round(Int, sum(sqrt((f.x2-f.x1)^2 + (f.y2-f.y1)^2) for f in fracs))
    println("  → $filename  |  $(nrow(df)) fractures  |  L_tot = $(Ltot) m  |  clusters = $n_cl")
end

# =============================================================================
#  MAIN
# =============================================================================
Random.seed!(SEED)
mkpath(OUTPUT_DIR)
outfile(regime) = joinpath(OUTPUT_DIR, "fractures_$(regime)_$(NETWORK_TAG).csv")

# base conjugate set
raw = []
append!(raw, gen_family(N_FRACS_PER_FAMILY, 1, +HALF_ANGLE))   # dips towards +x
append!(raw, gen_family(N_FRACS_PER_FAMILY, 2, -HALF_ANGLE))   # dips towards -x

# rescaling to the target total length
scale = L_TARGET / sum(f[4] for f in raw)
lens  = [f[4] * scale for f in raw]
println("Length scale factor = $(round(scale, digits=3)); ",
        "final lengths: $(round(minimum(lens), digits=1))–$(round(maximum(lens), digits=1)) m ",
        "(nominal range $(round(LMIN*scale, digits=1))–$(round(LMAX*scale, digits=1)) m)")

println("[1/2] Normal faulting")
nf = assign_clusters(build_network(raw, scale, NF_DIP, NF_STRIKE))
export_csv(nf, outfile("NF"))

println("[2/2] Thrust faulting (NF network rotated by 90°)")
tf = rotate_90(nf, TF_DIP, TF_STRIKE)
export_csv(tf, outfile("TF"))
