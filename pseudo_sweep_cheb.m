%written by Sebastian Acosta
%modified by Jesse Chan 24 Aug 2022
%modified by Raven Shane Johnson 16 Aug 2022

global c0 alpha OMEGA ORDER

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 25*pi;      % Angular frequency

%Determine accuracy
PPWx = 250;          % Points per wavelength in x-direction (marching)
ORDER = 2;          % Order of the Pade approximation

c0 = 1;                                             % reference wavespeed
alpha = 0.25;
lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength

Nx = round(PPWx * L / lambda);                      % N points in x-direction

dy = W / (Ny - 1);                                  % mesh size in y-direction
dx = L / (Nx - 1);                                  % mesh size in x-direction

% c = @(x, y) c0 * (1 - alpha * exp(-100 * ((x-0.5).^2 + y.^2)));
%dcdx = @(x,y) c0 * (alpha .* 200 .* (x - 0.5) .* exp(-100 * ((x-0.5).^2 + y.^2)));
% d_omega_invc2_dx = @(x,y) -2 * (OMEGA^2 ./ c(x,y).^3) .* dcdx(x,y);

fprintf('--------------------------------------------------- \n');
fprintf('Pseudo-diff Order = %i \n', ORDER);
fprintf('PPW x-axis = %.2g \n', lambda/dx);
fprintf('PPW y-axis = %.2g \n', lambda/dy);
fprintf('Nx x Ny = %i x %i \n', Nx, Ny);


%%

global D M A I invMA
N = 25;
[x, w] = JacobiGL(0, 0, N);
Ny = length(x);

V = Vandermonde1D(N, x);
D = GradVandermonde1D(N, x) / V;

% map to [-0.5, 0.5]
x = 0.5 * x;
w = 0.5 * w;
D = 2 * D;


M = spdiags(w, 0, Ny, Ny);
invM = spdiags(1 ./ w, 0, Ny, Ny);
I = speye(Ny, Ny);  % Identity matrix
A = D' * M * D;
invMA = invM * A;
Vplot = Vandermonde1D(N, linspace(-1, 1, 2*N)) / V;


% define space-time grid
t = linspace(0, L, Nx);
[x, y] = meshgrid(t, x);


%% time-stepping

C = c(x,y);
Cx = dcdx(x,y);
dinvC2 = d_omega_inv_c2(x,y);

f = @(x) (1 ./ c(x, y_FE).^2 - 1) .* OMEGA^2 .* exp(1i * OMEGA * x);
f = @(x) 0.0;

% sweep backwards with zero "final" condition
u = zeros(Ny, Nx);
dx_backwards = -dx;
for j=Nx:-1:2
    
    x1 = x(:,j);
    k1 = -DtN(u(:,j), C(:,j), dinvC2(:,j)) + f(x1);
    u1 = u(:,j) +  dx_backwards * k1;
    
    x2 = x1 + dx_backwards;
    k2 = -DtN(u1, C(:,j-1), dinvC2(:,j-1)) + f(x2);
    u2 = u1 + dx_backwards * k2;
    
    %Calculate next column of u
    u(:,j-1) = 0.5 * (u(:,j) + u2);
    
    %print current computation step
    if mod(j, 100) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
end

u_backwards = u;

% initial condition
u = zeros(Ny, Nx);
u(:, 1) = exp(1i .* y(:,1));

%MATRIX FREE EXPLICIT METHOD
for j=1:Nx-1
    
    x1 = x(:,j);
    k1 = DtN(u(:,j), C(:,j), dinvC2(:,j)) + u_backwards(:,j);
    u1 = u(:,j) + dx * k1;
    
    x2 = x1 + dx;
    k2 = DtN(u1, C(:,j+1), dinvC2(:,j+1)) + u_backwards(:,j+1);
    u2 = u1 + dx * k2;
    
    u(:,j+1) = 0.5 * (u(:,j) + u2);
    
    %print current computation step
    if mod(j, 100) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
    
end


%% PLOTS -----------------------------

figure
surf(Vplot * x, Vplot * y, real(Vplot * u)); hold on;
colormap copper;
axis image
shading interp;
hcolor = colorbar;
caxis([-1 1]);
view(0,90);
xlim([0 L]);
ylim([-W W]/2);
xticks(0:W/5:L);
yticks(-W/2:W/10:W/2);

h = gca;
h.FontSize = 10;

% %% compute DtN directly
%
% e = zeros(size(u(:,1)));
% DtN_matrix = zeros(length(e));
% for i = 1:length(e)
%     e(i) = 1;
%     DtN_matrix(:,i) = DtN(A_constant_k, x_sweeping(1), y_FE, a, b, e);
%     e(i) = 0;
% end
% DtN_matrix(abs(DtN_matrix) < 1e-10) = 0;


%%


function val = c(x,y)
global c0 alpha
val = c0 * (1 - alpha * exp(-25 * ((x-0.5).^2 + y.^2)));
end

function val = dcdx(x,y)
global c0 alpha
val = c0 * (alpha .* 50 .* (x - 0.5) .* exp(-25 * ((x-0.5).^2 + y.^2)));
end

function val = d_omega_inv_c2(x,y)
global OMEGA
val = -2 * (OMEGA^2 ./ c(x,y).^3) .* dcdx(x,y);
end

function D = spdiag(d)
global I
D = spdiags(d, 0, size(I,1), size(I,2));
end

%function for calculating DtN * u_j
function [u_next] = DtN(u, c, d_omega_inv_c2)

global OMEGA
global D M invMA I
global ORDER

k = OMEGA ./ c;
k_sq = k.^2;
inv_k_sq = (1 ./ k_sq);
A_k_sq = D' * M * spdiag(inv_k_sq) * D;
A_u = (A_k_sq * u) ./ diag(M);

[a_pade, b_pade] = pade_coefficients(ORDER);

lambda_1_u = u;
for j = 1:length(a_pade)
    lambda_1_u = lambda_1_u + (I + b_pade(j) .* A_k_sq) \ (a_pade(j) .* A_u);
end
lambda_1_u = 1i * k .* lambda_1_u; % mult by i*k \sum(...)

lambda_0_u = (spdiag(k_sq) + invMA) \ (-0.25 * d_omega_inv_c2 .* u);

u_next = lambda_1_u + lambda_0_u;

end
