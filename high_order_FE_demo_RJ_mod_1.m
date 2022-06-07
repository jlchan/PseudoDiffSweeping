% High order FE method on [-1,1]
    %written by Jesse Chan 2 June 2022
    %modified by Raven Shane Johnson 7 June 2022

K = 5; % number of elements
VX = linspace(-1, 1, K+1); % mesh vertices
h = diff(VX); %calculate node spacing

N = 7; % polynomial degree
[rq, wq] = JacobiGL(0, 0, N); % ref quadrature points and weights
[Vq, Vrq, r_interpolation] = lagrange_basis(N, rq); %calculate matrices with basis function inner product entries

%mass matrix on [-1,1] reference element
M = Vq' * diag(wq) * Vq;     

%introduce k(x) term
k = @(x) 1 + 0.5 * sin(pi*x);

%RHS function
f = @(x) exp(-100 * x.^2);

%assemble global matrices and RHS
total_num_nodes = ceil(N * K + 1);
A_global = sparse(total_num_nodes, total_num_nodes); %arrays of zeros to hold mass, stiffness, and RHS
M_global = sparse(total_num_nodes, total_num_nodes);
b = zeros(total_num_nodes, 1);
%loop over elements
for e = 1:K
    xq = VX(e) + (1 + rq) / 2 * h(e); %turning ref quad pts into og quad pts
    A = Vrq' * diag(wq.*k(xq)) * Vrq; %stiffness matrix on [-1,1] reference element
    starting_node_id = 1 + (e - 1) * N;
    node_ids = starting_node_id:(starting_node_id + N); %get global nodes for current element  
    %sum over elements to construct matrices and RHS
    A_global(node_ids, node_ids) = A_global(node_ids, node_ids) + 2 / h(e) * A;
    M_global(node_ids, node_ids) = M_global(node_ids, node_ids) + h(e) / 2 * M;
    b(node_ids) = b(node_ids) + h(e) / 2 * Vq' * (wq .* f(xq));
end

%calculate reference solution
u = (0.1 * M_global + A_global) \ b;

%extract the local solution on each element
x_local = zeros(N+1, K);
u_local = zeros(N+1, K);
for e = 1:K
    starting_node_id = 1 + (e - 1) * N;
    node_ids = starting_node_id:(starting_node_id + N);    
    u_local(:, e) = u(node_ids);
    x_local(:, e) = VX(e) + (1 + r_interpolation) / 2 * h(e);
end

%plot the function approximation
plot(x_local, u_local)

%look at structure of mass matrix
%tol = 1e-10
%M_global(abs(M_global)<tol) = 0
%spy(M_global)

%trust me on this one!
function [V, Vr, r_interpolation] = lagrange_basis(N, r)
    r_interpolation = JacobiGL(0, 0, N);
    VDM = Vandermonde1D(N, r_interpolation);
    V = Vandermonde1D(N, r) / VDM;
    Vr = GradVandermonde1D(N, r) / VDM;    
end

