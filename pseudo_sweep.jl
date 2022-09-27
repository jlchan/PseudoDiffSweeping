using LinearAlgebra, SparseArrays
using StartUpDG
using LinearAlgebra

peaks(x, y) = 3 * (1-x)^2 * exp(-(x.^2) - (y+1).^2) - 
              10 * (x / 5 - x^3 - y^5) * exp(-x^2 - y^2) - 
              1/3 * exp(-(x+1)^2 - y^2) 

function pade_coefficients(N)
    a = SVector{2}((2 / (2 * N + 1) * sin(i * pi / (2 * N + 1))^2 for i in 1:N))
    b = SVector{2}((cos(i * pi / (2 * N + 1))^2 for i in 1:N))
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

# # parameters for computing the Λ₁ pseudodifferential operator using a rational 
# # Pade approximation sqrt(A + B) ≈ ∑ (a_j * B * u) / (A + b_j * B) 
# # Usually, A = Diagonal(k^2) and B = M \ (FE Laplacian)
# struct Lambda_1{TA, TB, Ttmp}
#     ORDER::Int
#     A::TA
#     B::TB
#     tmp_storage::Ttmp
# end

# function (pade::Lambda_1)(u)
#     (; A, B, tmp_storage, ORDER) = pade
#     a_pade, b_pade = pade_coefficients(ORDER)
#     @. tmp_storage = u
#     for j in 1:length(a_pade)
#         tmp_storage .+= (A + b_pade[j] .* B) \ (a_pade[j] .* (B * u));
#     end
#     @. tmp_storage = 1im * k * tmp_storage # mult by i*k \sum(...)    
#     return tmp_storage 
# end

function apply_lambda_1(ORDER, invMA, k, u)    
    a_pade, b_pade = pade_coefficients(ORDER)
    lambda_1_u = copy(u)
    for j in 1:length(a_pade)
        lambda_1_u .+= (Diagonal(k.^2) + b_pade[j] .* invMA) \ (a_pade[j] .* (invMA * u));
    end
    return 1im * k .* lambda_1_u # mult by i*k \sum(...)    
end
    
# function for calculating DtN * u_j
function rhs!(du, u, p, t)
    OMEGA, ORDER, invMA = p
    k, d_inv_c2, d2_inv_c2 = p.c_data

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

function rhs(u, p, t) 
    du = fill!(similar(u), zero(eltype(u)))
    rhs!(du, u, p, t)
    return du
end
       
N = 3
num_elements = 8

OMEGA = 25 * pi   # Angular frequency
PPWx = 300        # Points per wavelength in x-direction (marching)
ORDER = 2         # Pseudo-diff order
#pseudodiff_params = (; OMEGA, PPWx, ORDER, c=(x,y) -> 1 - 0.25 * exp(-25 * ((x-0.5)^2 + y^2)))
pseudodiff_params = (; OMEGA, PPWx, ORDER, c=(x,y) -> 1 - .025 * peaks(10 * (x - 0.5), 10 * y))

(; OMEGA, PPWx, ORDER, c) = pseudodiff_params    

# create FE discretization
rd = RefElemData(Line(), N)
(VY,), EToV = uniform_mesh(Line(), num_elements)
md = MeshData(domain_height / 2 * VY, EToV, rd)                                           
M, A, y = assemble_FE_matrices(rd, md)
M_diagonal = Diagonal(vec(sum(M, dims=2)))
invMA = M_diagonal \ A
Ny = length(y)

# points in the sweeping direction
lambda = 2 * pi / OMEGA # reference wavelength
Nx = ceil(Int, PPWx * domain_width / lambda)  
dx = domain_width / (Nx - 1)
x = LinRange(0, domain_width, Nx)

(; c) = pseudodiff_params
d_inv_c2(x, y) = ForwardDiff.derivative(x -> 1 / c(x, y)^2, x)
d2_inv_c2(x, y) = ForwardDiff.derivative(x -> d_inv_c2(x, y), x)    

C = c.(x', y)
d_inv_C2 = d_inv_c2.(x', y)
d2_inv_C2 = d2_inv_c2.(x', y)
c_data = (; k=OMEGA ./ C[:,1], d_inv_c2=d_inv_C2[:,1], d2_inv_c2=d2_inv_C2[:,1])

# manufactured plane wave solution
u_exact(x, y) = exp(1im * OMEGA * x)
forcing = (x, y) -> (1 / c(x, y)^2 - 1) * OMEGA^2 * exp(1im * OMEGA * x) 
f = forcing.(x', y)
p = (; OMEGA, ORDER, invMA, C, d_inv_C2, d2_inv_C2, c_data)

function set_c_data!(c_data, p, i)
    (; C, d_inv_C2, d2_inv_C2) = p
    @. c_data.k = OMEGA ./ C[:, i]
    @. c_data.d_inv_c2 = d_inv_C2[:, i]
    @. c_data.d2_inv_c2 = d2_inv_C2[:, i]
end

# preallocate u, RHS
u = zeros(Complex{eltype(y)}, Ny, Nx)
u1, u2, du = ntuple(_ -> similar(u[:, 1]), 3)

# backwards sweep to "pick up" the forcing
for i in Nx:-1:2
    fill!(du, zero(eltype(du)))

    set_c_data!(c_data, p, i)
    rhs!(du, u[:,i], p, x[i]) 
    @. du = du - f[:,i]
    @. u1 = u[:,i] + dx * du

    set_c_data!(c_data, p, i-1)
    rhs!(du, u1, p, x[i-1]) 
    @. du = du - f[:,i-1]
    @. u2 = u1 + dx * du

    @. u[:,i-1] = 0.5 * (u[:,i] + u2)

    if i % 100 == 0
        println("On step $i out of $(Nx-1)")
    end
end

u_backwards = copy(u)

u[:,1] .= one(eltype(u))
for i in 1:Nx-1
    set_c_data!(c_data, p, i)
    rhs!(du, u[:,i], p, x[i])
    @. du = du + u_backwards[:, i]
    @. u1 = u[:,i] + dx * du

    set_c_data!(c_data, p, i+1)
    rhs!(du, u1, p, x[i+1]) 
    @. du = du + u_backwards[:, i+1]
    @. u2 = u1 + dx * du

    @. u[:, i+1] = 0.5 * (u[:, i] + u2)

    if i % 100 == 0
        println("On step $i out of $(Nx-1)")
    end
end

error = u - u_exact.(x', y)
w = repeat(diag(M_diagonal), 1, Nx)
w = domain_width * domain_height * w / sum(w) 
L2_error = sqrt(dot(w, @. abs(error)^2))

using Plots
gr(leg=false)
contourf(x, y, abs.(error), c=:viridis)
title!("L2 error = $L2_error")
