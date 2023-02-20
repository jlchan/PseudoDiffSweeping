using LinearAlgebra, SparseArrays
using StartUpDG
using LinearAlgebra
using ForwardDiff
using StaticArrays

# construct approximation of the tangential derivative operator
function assemble_FE_matrix(N, num_elements, domain_height; bc_type=:Neumann)
    rd = RefElemData(Line(), N)
    (VY,), EToV = uniform_mesh(Line(), num_elements)
    md = MeshData(domain_height / 2 * VY, EToV, rd)                                           

    M = spzeros(rd.N * md.num_elements + 1, rd.N * md.num_elements + 1)
    A = spzeros(rd.N * md.num_elements + 1, rd.N * md.num_elements + 1)
    x = zeros(rd.N * md.num_elements + 1)
    for e in 1:md.num_elements
        ids = (1:N+1) .+ (e-1) * rd.N        
        A[ids, ids] .+= 1.0 / md.J[1, e] * rd.Dr' * rd.M * rd.Dr
        M[ids, ids] .+= md.J[1, e] * rd.M
        x[ids] .= view(md.x, :, e)
    end

    @. A = -A # use d^2/dx^2 instead of -d^2/dx^2
    M_diagonal = Diagonal(vec(sum(M, dims=2))) # lumped mass matrix    

    return (M_diagonal \ A), x, diag(M_diagonal)
end
    
function pade_coefficients(N)
    a = SVector{N}((2 / (2 * N + 1) * sin(i * pi / (2 * N + 1))^2 for i in 1:N))
    b = SVector{N}((cos(i * pi / (2 * N + 1))^2 for i in 1:N))
    alpha = pi / 6 # this needs to be non-zero for stability
    e_scaling = exp(-im * alpha) - 1
    alpha_0 = exp(im * alpha / 2) * (1 + sum((a .* e_scaling) ./ (1 .+ b .* e_scaling)))
    a_alpha = (a * exp(-im * alpha / 2)) ./ (1 .+ b * e_scaling).^2
    b_alpha = (b * exp(-im * alpha)) ./ (1 .+ b * e_scaling)
    return alpha_0, a_alpha, b_alpha
end

# function for calculating DtN * u_j
function rhs!(du, u, cache, t)

    set_c_data!(cache, t)
    (; OMEGA, A) = cache
    k, c, inv_c2, d_inv_c2, d2_inv_c2 = cache.c_data
    
    fill!(du, zero(eltype(du)))
    
    # λ_1(u)
    alpha_0, a_pade, b_pade = cache.pade_coeffs
    Au = A * u
    @. du = alpha_0 * u
    for (a, b) in zip(a_pade, b_pade)        
        du .+= (Diagonal(k.^2) + b .* A) \ (a .* Au);
    end
    @. du *= 1im * k # mult by i*k \sum(...)

    # λ_0(u) = -1/4 * ∂(1/c^2)) / c^2 + O(1/ω^2)
    @. du += (-0.25 * d_inv_c2 ./ inv_c2 .* u)

    # λ_{-1}(u) = im * c / (8 * OMEGA) * (...) + O(1/ω^3)
    @. du += im * c / (8 * OMEGA) * (5/4 * (d_inv_c2 / inv_c2)^2 - d2_inv_c2 / inv_c2) * u
end

# create FE discretization    
N = 3
num_elements = 16
domain_width, domain_height = 1, 1
A, y, w1D = assemble_FE_matrix(N, num_elements, domain_height)

OMEGA = 40           # Angular frequency
PPWx  = 10            # Points per wavelength in x-direction (marching)
ORDER = 3            # Pseudo-diff order

Nx = ceil(Int, domain_width * OMEGA * PPWx)
Ny = length(y)
x = LinRange(-0.5, -0.5 + domain_width, Nx + 1)
dx = x[2] - x[1]

peaks(x, y) = 3 * (1-x)^2 * exp(-(x.^2) - (y+1).^2) - 
              10 * (x / 5 - x^3 - y^5) * exp(-x^2 - y^2) - 
              1/3 * exp(-(x+1)^2 - y^2) 

# one_minus_b = (x,y) -> 1 + 1.5 * exp(-160 * ((x-0.5)^2 + y^2))
b(x,y) = b(sqrt(x^2 + y^2)) # b should be given by "radial solution"
one_minus_b = (x,y) -> 1 - b(x,y) # OMEGA^2 * (1 - b(x)) <---> OMEGA^2 / c^2
c(x,y) = 1 / sqrt(one_minus_b(x, y)) # (1 - b(x)) = 1/c^2 ------> c = 1 / sqrt(1-b(x))
d_inv_c2(x, y) = ForwardDiff.derivative(x -> 1 / c(x, y)^2, x)
d2_inv_c2(x, y) = ForwardDiff.derivative(x -> d_inv_c2(x, y), x)    
c_data = (; k=similar(y), c = similar(y), inv_c2=similar(y), d_inv_c2=similar(y), d2_inv_c2=similar(y))

function set_c_data!(p, x)
    (; y, c_data) = p
    @. c_data.k = OMEGA ./ c(x, y)
    @. c_data.c = c(x, y)
    @. c_data.inv_c2 = 1 / c(x, y)^2
    @. c_data.d_inv_c2 = d_inv_c2(x, y)
    @. c_data.d2_inv_c2 = d2_inv_c2(x, y)
end

# manufactured plane wave solution
u_exact(x, y) = exp(1im * OMEGA * x)
dudx_exact(x, y) = ForwardDiff.derivative(x->u_exact(x, y), x)
dudy_exact(x, y) = ForwardDiff.derivative(y->u_exact(x, y), y)
laplacian_u_exact(x, y) = ForwardDiff.derivative(x->dudx_exact(x, y), x) + ForwardDiff.derivative(y->dudy_exact(x, y), y)

forcing = (x, y) -> (OMEGA^2  / c(x, y)^2) * u_exact(x, y) + laplacian_u_exact(x, y)
u_initial(x, y) = u_exact(x, y)
f = forcing.(x', y)

forcing = (x, y) -> 0 # no manufactured solution

cache = (; OMEGA, ORDER, A, c_data, y, pade_coeffs=pade_coefficients(ORDER))

# preallocate u, RHS
u = zeros(Complex{eltype(y)}, Ny, Nx + 1)
u_tmp, du, du_accum = ntuple(_ -> similar(u[:, 1]), 3)

function impose_homogeneous_BCs!(u)
    # u[1] = zero(eltype(u))
    # u[end] = zero(eltype(u))
end

if norm(forcing.(x', y)) < 100 * eps()
    println("Skipping backwards sweep - forcing is zero.")    
else
    # backwards sweep to "pick up" the forcing
    for i in Nx + 1:-1:2
        fill!(du_accum, zero(eltype(du_accum)))
        
        rhs!(du, u[:,i], cache, x[i]) 
        @. du = du - forcing(x[i], y)
        @. du_accum += du

        @. u_tmp = u[:,i] + 0.5 * dx * du
        impose_homogeneous_BCs!(u_tmp)
        rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
        @. du = du - forcing(x[i] - 0.5 * dx, y)
        @. du_accum += 2 * du

        @. u_tmp = u[:,i] + 0.5 * dx * du
        impose_homogeneous_BCs!(u_tmp)
        rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
        @. du = du - forcing(x[i] - 0.5 * dx, y)
        @. du_accum += 2 * du

        @. u_tmp = u[:,i] + dx * du
        impose_homogeneous_BCs!(u_tmp)
        rhs!(du, u_tmp, cache, x[i-1]) 
        @. du = du - forcing(x[i-1], y)
        @. du_accum += du
        
        @. u[:,i-1] = u[:,i] + (dx / 6) * du_accum
        impose_homogeneous_BCs!(view(u,:,i-1))
        if i % 100 == 0
            println("On step $i out of $(Nx-1)")
        end
    end
end

u_backwards = copy(u)

# temp memory for interpolation
u_backwards_mid = similar(u_backwards[:,1])
fill!(u, zero(eltype(u)))

function u_exact(x, y)
    θ = atan.(y, x)
    r = @. sqrt(x^2 + y^2)
    return sc(r, θ)
end
u_initial(x, y) = u_exact(x, y)

# should by symmetric anyways
u_left(t) = u_exact(t, 0.5)
u_right(t) = u_exact(t, -0.5)
function impose_BCs!(u, t)
    u[1] = u_left(t)
    u[end] = u_right(t)
end

u[:,1] .= u_initial.(x[1], y)
for i in 1:Nx

    if i < Nx - 2
        @. u_backwards_mid = 0.3125 * u_backwards[:, i] + 0.9375 * u_backwards[:, i+1] - 0.3125 * u_backwards[:, i+2] + 0.0625 * u_backwards[:, i+3]
    else
        @. u_backwards_mid = 0.3125 * u_backwards[:, i] + 0.9375 * u_backwards[:, i-1] - 0.3125 * u_backwards[:, i-2] + 0.0625 * u_backwards[:, i-3]
    end

    fill!(du_accum, zero(eltype(du_accum)))
    
    rhs!(du, u[:,i], cache, x[i]) 
    @. du = du + u_backwards[:, i]
    @. du_accum += du  

    @. u_tmp = u[:,i] + 0.5 * dx * du
    impose_BCs!(u_tmp, x[i] + 0.5 * dx)    
    rhs!(du, u_tmp, cache, x[i] + 0.5 * dx) 
    @. du = du + u_backwards_mid
    @. du_accum += 2 * du

    @. u_tmp = u[:,i] + 0.5 * dx * du
    impose_BCs!(u_tmp, x[i] + 0.5 * dx)
    rhs!(du, u_tmp, cache, x[i] + 0.5 * dx) 
    @. du = du + u_backwards_mid
    @. du_accum += 2 * du

    @. u_tmp = u[:,i] + dx * du
    impose_BCs!(u_tmp, x[i+1])
    rhs!(du, u_tmp, cache, x[i+1]) 
    @. du = du + u_backwards[:, i+1]
    @. du_accum += du 

    @. u[:,i+1] = u[:,i] + (dx / 6) * du_accum
    impose_BCs!(view(u, :, i+1), x[i+1])
    
    if i % 100 == 0
        println("On step $i out of $(Nx-1)")
    end
end

# u_forward = copy(u)

# # backwards sweep to "pick up" the forcing
# for i in Nx + 1:-1:2
#     fill!(du_accum, zero(eltype(du_accum)))
    
#     rhs!(du, u[:,i], cache, x[i]) 
#     @. du = du - u_forward[:,i]
#     @. du_accum += du

#     @. u_tmp = u[:,i] + 0.5 * dx * du
#     impose_BCs!(u_tmp, x[i] - 0.5 * dx)
#     rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
#     @. du = du - 0.5 * (u_forward[:,i] + u_forward[:,i-1])
#     @. du_accum += 2 * du

#     @. u_tmp = u[:,i] + 0.5 * dx * du
#     impose_BCs!(u_tmp, x[i] - 0.5 * dx)
#     rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
#     @. du = du - 0.5 * (u_forward[:,i] + u_forward[:,i-1])
#     @. du_accum += 2 * du

#     @. u_tmp = u[:,i] + dx * du
#     impose_BCs!(u_tmp, x[i-1])
#     rhs!(du, u_tmp, cache, x[i-1]) 
#     @. du = du - u_forward[:,i-1]
#     @. du_accum += du
    
#     @. u[:,i-1] = u[:,i] + (dx / 6) * du_accum
#     impose_BCs!(u_tmp, x[i-1])
#     if i % 100 == 0
#         println("On step $i out of $(Nx-1)")
#     end
# end

uex = u_exact.(x', y) 
uex[@. sqrt((x')^2 + y^2) < R] .= NaN
error = u - u_exact.(x', y)
w = repeat(w1D, 1, Nx+1)
w = domain_width * domain_height * w / sum(w) 
L2_error = sqrt(dot(w, @. abs(error)^2))
@show L2_error

# u_incident(x, y) = exp(1im * OMEGA * x)
# @. u -= u_incident(x', y)
# @. uex -= u_incident(x', y)

using Plots: Plots

p1 = Plots.contourf(x, y, real.(u), c=:viridis, clims=(-4, 4), leg=false, ratio=1, title="Sweeping")
p2 = Plots.contourf(x, y, real.(uex), c=:viridis,  clims=(-4, 4), leg=false, ratio=1, title="Analytical")
plot(p1, p2)

# Plots.contourf(x, y, abs.(u .- uex), c=:viridis, leg=false, ratio=1, colorbar=true, title="Error")
