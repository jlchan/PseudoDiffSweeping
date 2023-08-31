using SpecialFunctions
using OrdinaryDiffEq
using ForwardDiff

kappa = 50 * pi
r0 = 1e-12
R = .25 # b(R) is the error threshhold

b(r) = -1.5 * exp(-160 * (r)^2)

# u_l'' + (1 / r) * u_l' + [-l^2 / r^2 + (1 - b(r)) * kappa] * u_l = 0
#  u' = v
# -v' = (1 / r) * v + [-l^2 / r^2 + (1 - b(r)) * kappa] * u_l
# up = @(r,u) [u(2); (l^2./r.^2-om^2*(1-b(r)))*u(1) - u(2)./r];
function rhs_radial!(du, u, p, r)    
    (; l, b, kappa) = p
    u_l, v = u[1], u[2]
    du[1] = v 
    du[2] = -(v / r + (-l^2 / r^2 + (1 - b(r)) * kappa^2) * u_l)    
end

function compute_scattering_phase(l, kappa, r0, R) 
    # set up parameters and initial condition
    parameters = (; l, b, kappa)
    if l==0
        u0 = [1; 0]
    else
        u0 = [r0^l; l * r0^(l-1)] # u_l(r0), u_l'(r0)
    end
    
    # solve ODE 
    tspan = (r0, R)
    ode = ODEProblem(rhs_radial!, u0, tspan, parameters)
    sol = solve(ode, Vern7(); save_everystep=false, abstol=1e-14, reltol=1e-14)

    # compute scattering phases
    dhankelh1(nu, z) = ForwardDiff.derivative(z -> hankelh1(nu, z), z)
    beta_l = sol.u[end][2] / sol.u[end][1] # u'(R) / u(R)
    α_l = kappa * dhankelh1(l, kappa * R) - hankelh1(l, kappa * R) * beta_l
    a_l = -conj(α_l) / α_l
    return a_l
end

struct ScatteringExpansion{T1, T2}
    kappa::T1
    scattering_phases::T2
end

function ScatteringExpansion(kappa; L=40)
    a = zeros(ComplexF64, L+1)
    for l in 0:L
        r0 = 1e-10 
        if l > 0
            om0 = max(1, kappa * sqrt(abs(1-b(0))))
            Rturn = min(R, l / om0); # tweak to make growing mode dominate
            r0 = 0.8 * Rturn * sqrt(eps())^(1/l)
        end
        a[l+1] = compute_scattering_phase(l, kappa, r0, R)
    end
    return ScatteringExpansion(kappa, a)
end

function (f::ScatteringExpansion)(r, θ)
    (; kappa, scattering_phases) = f
    a = scattering_phases
    # val = 0.5 * (hankelh1(0, kappa * r) + a[1] * hankelh2(0, kappa * r))
    val = 0.5 * (a[1] - 1) * hankelh1(0, kappa * r)
    for l in 1:length(a)-1
        # val += im^l * (hankelh1(l, kappa * r) + a[l+1] * hankelh2(l, kappa * r)) * cos(l * θ)
        val += im^l * (a[l+1] - 1) * hankelh1(l, kappa * r) * cos(l * θ)
    end
    x = r * cos(θ)
    return val + exp(im * kappa * x)
end

sc = ScatteringExpansion(kappa; L=40)

# x = 1
# y = 0.5
# θ = atan.(y, x)
# r = @. sqrt(x^2 + y^2)
# u = sc(r, θ)

# using Plots: Plots

# xx = LinRange(-.5, .5, 250)
# yy = LinRange(-.5, .5, 250)
# xx = @. xx + 0 * yy'; 
# yy = @. 0 * xx + yy'; 
# θ = atan.(yy, xx)
# r = @. sqrt(xx^2 + y^2)
# u = sc.(r, θ)
# u[@. r < R] .= NaN
# Plots.contourf(xx, yy, real.(u)', leg=false, clims=(-2.5, 2.5), c=:viridis, ratio=1)

