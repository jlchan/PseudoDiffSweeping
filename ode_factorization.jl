using Plots
using StaticArrays
using ForwardDiff
using LinearAlgebra

struct RK4 end
struct Heun end

# ω = 100 * pi

# # d2p/dt2 + ω^2 / c^2 * p = f 
# p(t) = exp(im * ω * t)

c = t -> 1
c = t -> 1 - 0.25 * exp(-25 * (t-0.5).^2)
c = t -> 1 - 0.25 * sin(2 * pi * t)
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
    for i in Nsteps: -1: 2
        t_i = t[i]
        dv1 = rhs_forward(v[i], params, t_i) - f(t_i)
        dv2 = rhs_forward(v[i] + 0.5 * dt * dv1, params, t_i - 0.5 * dt) - f(t_i - 0.5 * dt)
        dv3 = rhs_forward(v[i] + 0.5 * dt * dv2, params, t_i - 0.5 * dt) - f(t_i - 0.5 * dt)
        dv4 = rhs_forward(v[i] + dt * dv3,       params, t_i - dt)       - f(t_i - dt)
        
        v[i-1] = v[i] + dt * (1/6) * (dv1 + 2*dv2 + 2*dv3 + dv4)        
    end

    vb = copy(v)
    # @show norm(vb)

    # forward sweep with backwards sweep vb as source
    v = zeros(ComplexF64, Nsteps)
    v[1] = 1.0 
    for i in 1:Nsteps-1
        t_i = t[i]

        # interpolate values: we rescale the time interval 
        #     i * dt ---- (i+1) * dt ---- (i+2) * dt ---- (i+3) * dt
        # to [-1, 1]
        #     -1      ----   -1/3    ----    1/3     ---- 1
        # To evaluate in between "i * dt ---- (i+1) * dt" is equivalent to evaluating at 
        # x_mid = 0.5 * (-1 - 1/3) on the rescaled interval [-1, 1]. We can interpolate 
        # by computing coefficients L_i(x_mid) where L_i are Lagrange polynomials. 
        # Then, we just multiply by the nodal coefficients vb[i:i+3]
        if i < Nsteps - 2
            vb_midpoint = dot(SVector(0.3125, 0.9375, -0.3125, 0.0625), vb[i:i+3])
        else    
            # if we are close to the last timestep, we flip the interpolation around. 
            vb_midpoint = dot(SVector(0.3125, 0.9375, -0.3125, 0.0625), vb[i:-1:i-3])
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
    return LinRange(1e-14, FinalTime, Nsteps+1)
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
    return maximum(abs.(v .- p.(parameters.t)))
end

omega = 25 * pi

ppw = 5:80
plot(inv.(omega * ppw), [compute_err(Heun(), omega, ppw) for ppw in ppw], xaxis=:log, yaxis=:log, label="RK2")

ppw = 1:40
plot!(inv.(omega * ppw), [compute_err(RK4(), omega, ppw) for ppw in ppw], xaxis=:log, yaxis=:log, label="RK4")