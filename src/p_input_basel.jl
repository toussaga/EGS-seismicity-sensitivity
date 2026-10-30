using Interpolations

data = CSV.read("Paper/WHP_basel.csv", DataFrame; delim=';')

# Extract columns (convert to Float64)
time_sec = Float64.(data[!, :t_s])
P_data   = Float64.(data[!, :P])      

# Interpolator
P_interp = LinearInterpolation(time_sec, P_data; extrapolation_bc=Flat())

# Boundary condition function
function simulate_real_pressure!(Pf, Tf, inj_points, time; T_inj=60.0+273.15)
    P_now = P_interp(time)                   # interpolated well pressure
    Pf = assign_P0(Pf, inj_points, P_now)    
    Tf = assign_P0(Tf, inj_points, T_inj)    
    return Pf, Tf
end