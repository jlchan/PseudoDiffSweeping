using LinearAlgebra, SparseArrays
using StartUpDG
using ForwardDiff
using StaticArrays
using Plots

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
num_elements = 128
domain_width, domain_height = 1, 1
A, y, w1D = assemble_FE_matrix(N, num_elements, domain_height)

kappa = 50 * pi

OMEGA = kappa       # Angular frequency, kappa defined in radial_solution.jl
PPWx  = 10           # Points per wavelength in x-direction (marching)
ORDER = 4           # Pseudo-diff order
num_iters = 2       # Number of sweeping iterations
interval = 500

Nx = ceil(Int, domain_width * OMEGA * PPWx)
Ny = length(y)
x = LinRange(-0.5, -0.5 + domain_width, Nx + 1)
dx = x[2] - x[1]

# 2D quadrature weights
w = repeat(w1D, 1, Nx+1)
w = domain_width * domain_height * w / sum(w) 

b(x,y) = b(sqrt(x^2 + y^2)) # b should be given by "radial_solution.jl"
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

forcing = (x, y) -> 0 # no manufactured solution (e.g., scattering)
f = forcing.(x', y)

# init as a plane wave 
u_initial(x, y) = exp(1im * OMEGA * x)

cache = (; OMEGA, ORDER, A, c_data, y, pade_coeffs=pade_coefficients(ORDER))

# # # 2nd order 2nd derivative matrix
# A_x = (1 / dx^2) * spdiagm(0 => -2 * ones(Nx + 1), 1 => ones(Nx), -1 => ones(Nx))
# A_x[1, 1:4] .= [2, -5, 4, -1] / dx^2
# A_x[end, end-3:end] .= [-1, 4, -5, 2] / dx^2

# 4th order 2nd derivative matrix - to match the order of RK4
A_x = (1 / dx^2) * spdiagm(0 => -5/2 * ones(Nx + 1), 
                           1 => 4/3 * ones(Nx),  2 => -1/12 * ones(Nx-1), 
                          -1 => 4/3 * ones(Nx), -2 => -1/12 * ones(Nx-1))                          
A_x[1, 1:6] .= [15/4, -77/6, 107/6, -13, 61/12, -5/6] / dx^2
A_x[2, 1:7] .= [0, 15/4, -77/6, 107/6, -13, 61/12, -5/6] / dx^2
A_x[end-1, end-6:end] .= reverse([0, 15/4, -77/6, 107/6, -13, 61/12, -5/6] ) / dx^2
A_x[end, end-5:end] .= reverse([15/4, -77/6, 107/6, -13, 61/12, -5/6]) / dx^2

function impose_homogeneous_BCs!(u, args...)
    # u[1] = zero(eltype(u))
    # u[end] = zero(eltype(u))
end

function impose_BCs!(delta_u, u, t)
end

# preallocate u, v, RHS
u = zeros(Complex{eltype(y)}, Ny, Nx + 1)
v = similar(u)

# local storage over a single time slice
u_tmp, du, du_accum = ntuple(_ -> similar(u[:, 1]), 3)

# temp memory for interpolation
r_mid, v_mid = similar(u[:, 1]), similar(u[:,1])

# "background" solution u
u_init = ((x, y) -> exp(1im * OMEGA * x)).(x', y)
fill!(u, zero(eltype(u)))

# # !!! TESTING
# Hu_init = (A * u_init + u_init * A_x') #+ (@. (OMEGA^2 / c(x', y)^2) * u_init)
# Hu_exact = d2udx2_exact.(x', y) + d2udy2_exact.(x', y) #+ (@. (OMEGA^2 / c(x', y)^2) * u_init)

# Hu_init = (A * (u_init) + (u_init) * A_x') + (@. (OMEGA^2 / c(x', y)^2) * (u_init))
# residual = f - Hu_init
# residual_norm = sum(w .* abs2.(residual))
# @show residual_norm

for i = 1:num_iters

    #println("On iteration $i")

    # compute (Δu + ω^2 / c^2 * u)
    global u_init = u + u_init;
    fill!(v, zero(eltype(v)))
    #           d2u/dy2       d2u/dx2

    Hu_init = (A * (u_init) + (u_init) * A_x') + (@. (OMEGA^2 / c(x', y)^2) * (u_init))
    residual = f - Hu_init

    # backwards sweep to "pick up" the forcing
    for i in Nx + 1:-1:2
        fill!(du_accum, zero(eltype(du_accum)))

        # interpolate to midpoint between x[i-1] and x[i]
        if i > 3 # < Nx - 2
            @. r_mid = 0.0625 * residual[:, i-3] - 0.3125 * residual[:, i-2] + 0.9375 * residual[:, i-1] + 0.3125 * residual[:, i]
        else # if i==2
            @. r_mid = 0.3125 * residual[:, i-1] + 0.9375  * residual[:, i] - 0.3125 * residual[:, i+1] +  0.0625 * residual[:, i+2]
        end
        
        rhs!(du, v[:,i], cache, x[i]) 
        @. du = du - residual[:, i] 
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += du

        @. u_tmp = v[:,i] + 0.5 * dx * du
        # impose_homogeneous_BCs!(u_tmp)
        rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
        @. du = du - r_mid
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += 2 * du

        @. u_tmp = v[:,i] + 0.5 * dx * du
        # impose_homogeneous_BCs!(u_tmp)
        rhs!(du, u_tmp, cache, x[i] - 0.5 * dx) 
        @. du = du - r_mid
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += 2 * du

        @. u_tmp = v[:,i] + dx * du
        # impose_homogeneous_BCs!(u_tmp)
        rhs!(du, u_tmp, cache, x[i-1]) 
        @. du = du - residual[:, i-1]
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += du
        
        @. v[:,i-1] = v[:,i] + (dx / 6) * du_accum
        # impose_homogeneous_BCs!(view(v, :, i-1))
        if i % interval == 0
            #println("On step $i out of $(Nx-1)")
        end
    end

    # forward sweep to incorporate v
    for i in 1:Nx

        # interpolate to midpoint between x[i] and x[i+1]
        if i < Nx - 2
            @. v_mid = 0.3125 * v[:, i] + 0.9375 * v[:, i+1] - 0.3125 * v[:, i+2] + 0.0625 * v[:, i+3]
            u_init_mid = 0.3125 * u_init[:, i] + 0.9375 * u_init[:, i+1] - 0.3125 * u_init[:, i+2] + 0.0625 * u_init[:, i+3]            
        else # if i == Nx-1 or Nx
            @. v_mid = 0.0625 * v[:, i-2] - 0.3125 * v[:, i-1] + 0.9375 * v[:, i] + 0.3125 * v[:, i+1]
            u_init_mid = 0.0625 * u_init[:, i-2] - 0.3125 * u_init[:, i-1] + 0.9375 * u_init[:, i] + 0.3125 * u_init[:, i+1]
        end

        fill!(du_accum, zero(eltype(du_accum)))
        
        rhs!(du, u[:,i], cache, x[i]) 
        @. du = du + v[:, i]
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += du  

        @. u_tmp = u[:,i] + 0.5 * dx * du
        impose_BCs!(u_tmp, u_init_mid, x[i] + 0.5 * dx)    
        rhs!(du, u_tmp, cache, x[i] + 0.5 * dx) 
        @. du = du + v_mid
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += 2 * du

        @. u_tmp = u[:,i] + 0.5 * dx * du
        impose_BCs!(u_tmp, u_init_mid, x[i] + 0.5 * dx)
        rhs!(du, u_tmp, cache, x[i] + 0.5 * dx) 
        @. du = du + v_mid
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += 2 * du

        @. u_tmp = u[:,i] + dx * du
        impose_BCs!(u_tmp, view(u_init, :, i+1), x[i+1])
        rhs!(du, u_tmp, cache, x[i+1]) 
        @. du = du + v[:, i+1]
        # du[1,:] .= 0; du[end,:] .= 0; 
        @. du_accum += du 
        
        @. u[:,i+1] = u[:,i] + (dx / 6) * du_accum
        impose_BCs!(view(u, :, i+1), view(u_init, :, i+1), x[i+1])
        
        if i % interval == 0
            #println("On step $i out of $(Nx-1)")
        end
    end

    Hu_init = (A * (u + u_init) + (u + u_init) * A_x') + (@. (OMEGA^2 / c(x', y)^2) * (u + u_init))
    residual = f - Hu_init
    residual[1:2,:] .= 0
    residual[end-1:end,:] .= 0
    residual_norm = sqrt(sum(w .* abs2.(residual)))
    @show residual_norm

end

Plots.contourf(x, y, real.(u_init + u), c=:viridis, leg=false, ratio=1)

# p1 = Plots.contourf(x, y, real.(u_init + u), c=:viridis, leg=false, ratio=1, title="Sweeping", clims=(-2.5, 2.5))
#p1 = Plots.contourf(x, y, real.(u_init), c=:viridis, clims=(-4, 4), leg=false, ratio=1, title="Sweeping")
#p2 = Plots.contourf(x, y, real.(uex), c=:viridis, clims=(-2.5, 2.5), leg=false, ratio=1, title="Analytical")
#Plots.plot(p1, p2)

#Plots.contourf(x, y, abs.(uex-(u_init + u)), c=:viridis, leg=false, ratio=1, title="Error", colorbar=true)

