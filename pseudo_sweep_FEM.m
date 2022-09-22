%written by Sebastian Acosta
%modified by Jesse Chan 24 Aug 2022
%modified by Raven Shane Johnson 16 Aug 2022

global c0 alpha OMEGA ORDER

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 100*pi;      % Angular frequency

%Determine accuracy
PPWx = 150;          % Points per wavelength in x-direction (marching)
ORDER = 2;          % Order of the Pade approximation

c0 = 1;                                             % reference wavespeed
alpha = 0.25;
lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength

Nx = round(PPWx * L / lambda);                      % N points in x-direction
dx = L / (Nx - 1);                                  % mesh size in x-direction

fprintf('--------------------------------------------------- \n');
fprintf('Pseudo-diff Order = %i \n', ORDER);
fprintf('PPW x-axis = %.2g \n', lambda/dx);


%% build FE matrices

global invMA Ny

N = 4;
VX = linspace(-0.5, 0.5, 8);
[M, A, r, global_to_local_ids] = compute_FE_matrices(N, VX);

Ny = length(r); % r = one-dimensional finite element points 

I = speye(Ny, Ny);
invMA = M \ A;

% construct space-time grids
t = linspace(0, L, Nx);
[x, y] = meshgrid(t, r);


%% time-stepping

C = c(x, y);
Cx = dcdx(x, y);
dinvC2 = d_inv_c2(x, y);
f = (1 ./ C.^2 - 1) .* OMEGA^2 .* exp(1i * OMEGA * x); % forcing for manufactured solution

% sweep backwards with zero "final" condition
u = zeros(Ny, Nx);
dx_backwards = -dx;
for j=Nx:-1:2
    
    x1 = x(:,j);
    k1 = -rhs(u(:,j), C(:,j), dinvC2(:,j)) + f(:,j);
    u1 = u(:,j) +  dx_backwards * k1;
    
    x2 = x1 + dx_backwards;
    k2 = -rhs(u1, C(:,j-1), dinvC2(:,j-1)) + f(:,j-1);
    u2 = u1 + dx_backwards * k2;
    
    %Calculate next column of u
    u(:,j-1) = 0.5 * (u(:,j) + u2);
    
    %print current computation step
    if mod(j, 500) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
end

u_backwards = u;

% initial condition
u = zeros(Ny, Nx);
u(:, 1) = 1;

%MATRIX FREE EXPLICIT METHOD
for j=1:Nx-1
    
    x1 = x(:,j);
    k1 = rhs(u(:,j), C(:,j), dinvC2(:,j)) + u_backwards(:,j);
    u1 = u(:,j) + dx * k1;
    
    x2 = x1 + dx;
    k2 = rhs(u1, C(:,j+1), dinvC2(:,j+1)) + u_backwards(:,j+1);
    u2 = u1 + dx * k2;
    
    u(:,j+1) = 0.5 * (u(:,j) + u2);
    
    %print current computation step
    if mod(j, 500) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
    
end


%% PLOTS -----------------------------

figure
uex = exp(1i * OMEGA * x);
err = abs(u - uex);

surf(x, y, real(u));
w = repmat(full(diag(M)), 1, size(x, 2));
w = w / sum(w(:));

L2_error = sum(sum(w .* err.^2));
title(['L2 error = ' num2str(L2_error)])
colormap copper; axis image; shading interp; hcolor = colorbar;
view(0,90); xlim([0 L]); ylim([-W W]/2);
xticks(0:W/5:L); yticks(-W/2:W/10:W/2);

h = gca;
h.FontSize = 10;

%%


function val = c(x,y)
global c0 alpha
a = 25;
val = c0 * (1 - alpha * exp(-a * ((x-0.5).^2 + y.^2)));
end

function val = dcdx(x,y)
global c0 alpha
a = 25;
val = c0 * (alpha .* 2 * a .* (x - 0.5) .* exp(-a * ((x-0.5).^2 + y.^2)));
end

function val = d_inv_c2(x,y)
val = -2 * (dcdx(x,y) ./ c(x,y).^3);
end

function D = spdiag(d)
global Ny
D = spdiags(d, 0, Ny, Ny);
end

function lambda_1_u = apply_lambda_1(k, u)
global invMA ORDER

[a_pade, b_pade] = pade_coefficients(ORDER);
lambda_1_u = u;
for j = 1:length(a_pade)
    lambda_1_u = lambda_1_u + (spdiag(k.^2) + b_pade(j) .* invMA) \ (a_pade(j) .* (invMA * u));
end
lambda_1_u = 1i * k .* lambda_1_u; % mult by i*k \sum(...)

end

%function for calculating DtN * u_j
function dudt = rhs(u, c, d_inv_c2)

global OMEGA
global invMA

k = OMEGA ./ c;
lambda_1_u = apply_lambda_1(k, u);
lambda_0_u = (spdiag(k.^2) + invMA) \ (-0.25 * OMEGA^2 * d_inv_c2 .* u);

tmp_1 = -0.25 * ((spdiag(k.^2) + invMA) * (OMEGA^2 * d2_inv_c2 .* u) - (OMEGA^2 * d_inv_c2).^2 .* u);
tmp_2 = (0.25 * OMEGA^2 * d2_inv_c2).^2 .* u;
lambda_n1_u = 

dudt = lambda_1_u + lambda_0_u;

end
