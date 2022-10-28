using LinearAlgebra, SparseArrays
using StartUpDG
using LinearAlgebra
using ForwardDiff
using StaticArrays

peaks(x, y) = 3 * (1-x)^2 * exp(-(x.^2) - (y+1).^2) - 
              10 * (x / 5 - x^3 - y^5) * exp(-x^2 - y^2) - 
              1/3 * exp(-(x+1)^2 - y^2) 

function pade_coefficients(N)
    a = SVector{N}((2 / (2 * N + 1) * sin(i * pi / (2 * N + 1))^2 for i in 1:N))
    b = SVector{N}((cos(i * pi / (2 * N + 1))^2 for i in 1:N))
    return a, b
end

function assemble_FE_matrices(rd, md; diffusivity_function = x -> 1.0)
    Vrq = rd.Vq * rd.Dr
    diffusivity = diffusivity_function.(md.xq)
    M = spzeros(rd.N * md.num_elements + 1, rd.N * md.num_elements + 1)
    A = spzeros(rd.N * md.num_elements + 1, rd.N * md.num_elements + 1)
    x = zeros(rd.N * md.num_elements + 1)
    for e in 1:md.num_elements
        ids = (1:N+1) .+ (e-1) * rd.N        
        A[ids, ids] .+= 1.0 / md.J[1, e] * Vrq' * diagm(rd.wq .* view(diffusivity, :, e)) * Vrq
        M[ids, ids] .+= md.J[1, e] * rd.M
        x[ids] .= view(md.x, :, e)
    end
    return M, A, x
end

function apply_lambda_1(ORDER, invMA, k, u)    
    a_pade, b_pade = pade_coefficients(ORDER)
    lambda_1_u = copy(u)
    for (a, b) in zip(a_pade, b_pade)
        lambda_1_u .+= (Diagonal(k.^2) + b .* invMA) \ (a .* (invMA * u));
    end
    return 1im * k .* lambda_1_u # mult by i*k \sum(...)    
end
    
# function for calculating DtN * u_j
function rhs!(du, u, cache, t)

    set_c_data!(cache, t)

    OMEGA, ORDER, invMA = cache
    k, d_inv_c2, d2_inv_c2 = cache.c_data

    k_sq_plus_invMA = lu(Diagonal(k.^2) + invMA) # TODO: rewrite as Diagonal(k.^2 .* w) + A and use `cholesky`

    lambda_1_u = apply_lambda_1(ORDER, invMA, k, u)
    lambda_0_u = k_sq_plus_invMA \ (-0.25 * OMEGA^2 * d_inv_c2 .* u)

    tmp_1 = -0.25 * ((Diagonal(k.^2) + invMA) * (OMEGA^2 * d2_inv_c2 .* u) - (OMEGA^2 * d_inv_c2).^2 .* u)
    tmp_1 .= k_sq_plus_invMA \ (k_sq_plus_invMA \ tmp_1)
    tmp_2 = k_sq_plus_invMA \ (k_sq_plus_invMA \ ((0.25 * OMEGA^2 * d_inv_c2).^2 .* u)) 
    @. tmp_2 = 0.5 * (tmp_1 + tmp_2)
    lambda_n1_u = k_sq_plus_invMA \ apply_lambda_1(ORDER, invMA, k, tmp_2)

    @. du = lambda_1_u + lambda_0_u + lambda_n1_u
end
       
N = 3
num_elements = 16

OMEGA = 40 * pi         # Angular frequency
PPWx = 10         # Points per wavelength in x-direction (marching)
ORDER = 3         # Pseudo-diff order
# pseudodiff_params = (; OMEGA, PPWx, ORDER, c=(x,y) -> 1 - 0.25 * exp(-25 * ((x-0.5)^2 + y^2)))
# kappa = (x,y) -> 1 + 1.5 * exp(-160 * ((x-0.5)^2 + y^2)) # kappa^2 * (1 - b(x)) <---> OMEGA^2 / c^2
# smoothed_step(x) = 0.5 * (tanh(25^2 * (x + .05)) + tanh(-25^2 * (x-.05)))
# kappa = (x,y) -> 1 + 4 * smoothed_step((x-0.5)^2 + y^2) # kappa^2 * (1 - b(x)) <---> OMEGA^2 / c^2
kappa = (x,y) -> 1 + 1.5 * exp(-160 * ((x-0.5)^2 + y^2)) # kappa^2 * (1 - b(x)) <---> OMEGA^2 / c^2
pseudodiff_params = (; OMEGA, PPWx, ORDER, c=(x,y) -> 1 / sqrt(kappa(x, y)))
# pseudodiff_params = (; OMEGA, PPWx, ORDER, c=(t,y) -> 1 - 0.25 * exp(-25 * (t-0.5).^2))
# pseudodiff_params = (; OMEGA, PPWx, ORDER, c=(x,y) -> 1)

(; OMEGA, PPWx, ORDER, c) = pseudodiff_params    

# create FE discretization
rd = RefElemData(Line(), N)
(VY,), EToV = uniform_mesh(Line(), num_elements)
domain_width, domain_height = 1, 1
md = MeshData(domain_height / 2 * VY, EToV, rd)                                           
M, A, y = assemble_FE_matrices(rd, md)
M_diagonal = Diagonal(vec(sum(M, dims=2))) # lumped mass matrix
invMA = M_diagonal \ A
Ny = length(y)

dx = 1 / (OMEGA * PPWx)
Nx = ceil(Int, domain_width / dx)
x = LinRange(1e-14, domain_width, Nx + 1)
dx = x[2] - x[1]

(; c) = pseudodiff_params
d_inv_c2(x, y) = ForwardDiff.derivative(x -> 1 / c(x, y)^2, x)
d2_inv_c2(x, y) = ForwardDiff.derivative(x -> d_inv_c2(x, y), x)    

c_data = (; k=similar(y), d_inv_c2=similar(y), d2_inv_c2=similar(y))

# manufactured plane wave solution
u_exact(x, y) = exp(1im * OMEGA * x)
forcing = (x, y) -> (1 / c(x, y)^2 - 1) * OMEGA^2 * exp(1im * OMEGA * x) 
u_initial(x, y) = u_exact(x, y)
f = forcing.(x', y)

forcing = (x, y) -> 0 # no manufactured solution

cache = (; OMEGA, ORDER, invMA, c_data, y)

function set_c_data!(p, x)
    (; y, c_data) = p
    @. c_data.k = OMEGA ./ c(x, y)
    @. c_data.d_inv_c2 = d_inv_c2(x, y)
    @. c_data.d2_inv_c2 = d2_inv_c2(x, y)
end

# preallocate u, RHS
u = zeros(Complex{eltype(y)}, Ny, Nx + 1)
u_tmp, du, du_accum = ntuple(_ -> similar(u[:, 1]), 3)

# backwards sweep to "pick up" the forcing
for i in Nx + 1:-1:2
    fill!(du_accum, zero(eltype(du_accum)))
    
    rhs!(du, u[:,i], cache, x[i]) 
    @. du = du - forcing(x[i], y)
    @. du_accum += du

    @. u_tmp = u[:,i] + 0.5 * dx * du
    rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
    @. du = du - forcing(x[i] - 0.5 * dx, y)
    @. du_accum += 2 * du

    @. u_tmp = u[:,i] + 0.5 * dx * du
    rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
    @. du = du - forcing(x[i] - 0.5 * dx, y)
    @. du_accum += 2 * du

    @. u_tmp = u[:,i] + dx * du
    rhs!(du, u_tmp, cache, x[i-1]) 
    @. du = du - forcing(x[i-1], y)
    @. du_accum += du
    
    @. u[:,i-1] = u[:,i] + (dx / 6) * du_accum

    if i % 100 == 0
        println("On step $i out of $(Nx-1)")
    end
end

u_backwards = copy(u)

# temp memory for interpolation
u_backwards_mid = similar(u_backwards[:,1])

fill!(u, zero(eltype(u)))
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
    rhs!(du, u_tmp, cache, x[i] + 0.5 * dx) 
    @. du = du + u_backwards_mid
    @. du_accum += 2 * du

    @. u_tmp = u[:,i] + 0.5 * dx * du
    rhs!(du, u_tmp, cache, x[i] + 0.5 * dx) 
    @. du = du + u_backwards_mid
    @. du_accum += 2 * du

    @. u_tmp = u[:,i] + dx * du
    rhs!(du, u_tmp, cache, x[i+1]) 
    @. du = du + u_backwards[:, i+1]
    @. du_accum += du 

    @. u[:,i+1] = u[:,i] + (dx / 6) * du_accum

    if i % 100 == 0
        println("On step $i out of $(Nx-1)")
    end
end

error = u - u_exact.(x', y)
w = repeat(diag(M_diagonal), 1, Nx+1)
w = domain_width * domain_height * w / sum(w) 
L2_error = sqrt(dot(w, @. abs(error)^2))

# using Plots
# gr(leg=false)
# GLMakie.contourf(x, y, abs.(error), c=:viridis, leg=false)
# title!("L2 error = $L2_error")
contourf(x, y, real.(u), c=:viridis, leg=false, ratio=1)