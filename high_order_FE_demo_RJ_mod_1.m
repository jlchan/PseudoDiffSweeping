% High order FE method on [-1,1]
%written by Jesse Chan 2 June 2022
%modified by Raven Shane Johnson 7 June 2022

N = 7; % polynomial degree
K = 5; 
VX = linspace(-1, 1, K+1); % mesh vertices
k = @(x) 1 + 0.5 * sin(pi*x); % introduce k(x) term
f = @(x) exp(-100 * x.^2); % RHS function
[M, A, b, x, global_to_local_ids] = compute_FE_system(N, VX, k, f);

u = (M + A) \ b;

u_local = u(global_to_local_ids);
x_local = x(global_to_local_ids);
r_plot = linspace(-1, 1, 100);
[Vplot, ~] = lagrange_basis(N, r_plot);
plot(Vplot * x_local, Vplot * u_local)

