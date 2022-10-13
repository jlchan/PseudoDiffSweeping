using Plots
using StaticArrays
using ForwardDiff

ω = 40 * pi
ppw = 80

dt = 1 / (ω * ppw)

FinalTime = 1.0
Nsteps = ceil(Int, FinalTime / dt)
dt = FinalTime / Nsteps
t = LinRange(1e-14, FinalTime, Nsteps)

# d2p/dt2 + ω^2 / c^2 * p = f 
p(t) = exp(im * ω * t)
# c = t -> 1
# c = t -> 1 - 0.25 * exp(-25 * (t-0.5).^2)
c = t -> 1 - 0.25 * sin(2 * pi * t)
# d_inv_c_dt(t) = ForwardDiff.derivative(t -> 1 / c(t), t)
dcdt(t) = ForwardDiff.derivative(t -> c(t), t)
d2cdt2(t) = ForwardDiff.derivative(dcdt, t)

function rhs_forward(v, p, t) 
    (; ω, c) = p
    Λ₁⁻  = -im * ω / c(t)
    Λ₀⁻  = -dcdt(t) / (2 * c(t)) 
    Λ₋₁⁻ = c(t) / (2 * im * ω) * (d2cdt2(t) / (2 * c(t)) - dcdt(t)^2 / (4 * c(t)^2))    
    # Λ₋₁⁻ = 0
    return -(Λ₁⁻ + Λ₀⁻ + Λ₋₁⁻) * v
end

rhs_backward(v, p, t) = -rhs_forward(v, p, t)

function sweep(::Heun, params)
    (; f, t) = params
    dt = t[2] - t[1]

    Nsteps = length(t)

    v = zeros(ComplexF64, Nsteps)

    # backwards sweep with zero final BC
    v[end] = zero(ComplexF64)
    dt_backwards = -dt
    for i in Nsteps:-1:2
        dv1 = rhs_backward(v[i], params, t[i]) + f(t[i])
        v1 = v[i] + dt_backwards * dv1
        dv2 = rhs_backward(v1, params, t[i-1]) + f(t[i-1])    
        v2 = v1 + dt_backwards * dv2
        v[i-1] = 0.5 * (v[i] + v2)
    end

    vb = copy(v)

    # forward sweep with backwards sweep vb as source
    v[1] = 1 # p(0.0) # assumes p(0) = exp(im * ω * 0) = 1
    for i in 1:Nsteps-1
        dv1 = rhs_forward(v[i], params, t[i]) + vb[i]
        v1 = v[i] + dt * dv1
        dv2 = rhs_forward(v1, params, t[i+1]) + vb[i+1]
        v2 = v1 + dt * dv2
        v[i+1] = 0.5 * (v[i] + v2)
    end
    return v
end


function sweep(::RK4, params)
    (; f, t) = params
    dt = t[2] - t[1]
    Nsteps = length(t)

    v = zeros(ComplexF64, Nsteps)

    # backwards sweep with zero final BC
    v[end] = zero(ComplexF64)
    dt_backwards = -dt
    t_i = t[end]
    for i in Nsteps: -1: 2
        dv1 = rhs_backward(v[i], params, t_i) + f(t_i)
        dv2 = rhs_backward(v[i] + 0.5 * dt_backwards * dv1, params, t_i + 0.5 * dt_backwards) + f(t_i + 0.5 * dt_backwards)
        dv3 = rhs_backward(v[i] + 0.5 * dt_backwards * dv2, params, t_i + 0.5 * dt_backwards) + f(t_i + 0.5 * dt_backwards)
        dv4 = rhs_backward(v[i] + dt_backwards * dv3,       params, t_i + dt_backwards)       + f(t_i + dt_backwards)
        
        v[i-1] = v[i] + dt_backwards * (1/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)

        t_i += dt_backwards
    end

    vb = copy(v)

    # forward sweep with backwards sweep vb as source
    v = zeros(ComplexF64, Nsteps)
    v[1] = 1.0 
    for i in 1:Nsteps-1
        t_i = t[i]

        # interpolate values 
        if i < Nsteps - 2
            vb_midpoint = dot(SVector(0.3125, 0.9375, -0.3125, 0.0625), vb[i:i+3])
        else
            vb_midpoint = dot(SVector(0.0625, -0.3125, 0.9375, 0.3125), vb[i-3:i])
        end

        dv1 = rhs_forward(v[i], params, t_i) + vb[i]
        dv2 = rhs_forward(v[i] + 0.5 * dt * dv1, params, t_i + 0.5 * dt) + vb_midpoint
        dv3 = rhs_forward(v[i] + 0.5 * dt * dv2, params, t_i + 0.5 * dt) + vb_midpoint
        dv4 = rhs_forward(v[i] + dt * dv3,       params, t_i + dt)       + vb[i+1]

        v[i+1] = v[i] + dt * (1/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)
    end
    return v
end

function t_ppw(ω, ppw)
    FinalTime = 1.0
    dt = 1 / (ω * ppw)
    Nsteps = ceil(Int, FinalTime / dt)
    dt = FinalTime / Nsteps
    return LinRange(1e-14, FinalTime, Nsteps + 1)
end

function setup(ω, ppw)
    t = t_ppw(ω, ppw)
    p(t) = exp(im * ω * t)
    dpdt(t) = ForwardDiff.derivative(p, t)
    d2pdt2(t) = ForwardDiff.derivative(dpdt, t)
    f = t -> d2pdt2(t) + ω^2 / c(t)^2 * p(t)
    parameters = (; ω, c, f, t)
    return p, parameters
end

function compute_err(method, ω, ppw) 
    p, parameters = setup(ω, ppw)
    v = sweep(method, parameters)
    return maximum(abs.(v - p.(parameters.t)))    
end
