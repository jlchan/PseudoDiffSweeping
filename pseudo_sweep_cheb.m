%written by Sebastian Acosta
%modified by Jesse Chan 24 Aug 2022
%modified by Raven Shane Johnson 16 Aug 2022

global c0 alpha OMEGA ORDER

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 25*pi;      % Angular frequency

%Determine accuracy
PPWx = 200;          % Points per wavelength in x-direction (marching)
ORDER = 2;          % Order of the Pade approximation

c0 = 1;                                             % reference wavespeed
alpha = 0.25;
lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength

Nx = round(PPWx * L / lambda);                      % N points in x-direction

dx = L / (Nx - 1);                                  % mesh size in x-direction

% c = @(x, y) c0 * (1 - alpha * exp(-100 * ((x-0.5).^2 + y.^2)));
%dcdx = @(x,y) c0 * (alpha .* 200 .* (x - 0.5) .* exp(-100 * ((x-0.5).^2 + y.^2)));
% d_omega_invc2_dx = @(x,y) -2 * (OMEGA^2 ./ c(x,y).^3) .* dcdx(x,y);

fprintf('--------------------------------------------------- \n');
fprintf('Pseudo-diff Order = %i \n', ORDER);
fprintf('PPW x-axis = %.2g \n', lambda/dx);


%% build FE matrices

global D M A I invMA Vq wq
N = 25;
[r, w] = JacobiGL(0, 0, N);
[rq, wq] = JacobiGL(0, 0, N);
Ny = length(r);

I = speye(Ny, Ny);
V = Vandermonde1D(N, r);
D = GradVandermonde1D(N, r) / V;
Vq = Vandermonde1D(N, rq) / V;

% map to [-0.5, 0.5]
r = 0.5 * r;
rq = 0.5 * rq;
wq = 0.5 * wq;
D = 2 * D;

M = spdiag(wq);
A = D' * M * D;
invMA = M \ A;
Vplot = Vandermonde1D(N, linspace(-1, 1, 2*N)) / V;

% construct space-time grids
t = linspace(0, L, Nx);
[x, y] = meshgrid(t, r);
[~, yq] = meshgrid(t, rq);


%% time-stepping

C = c(x, yq);
Cx = dcdx(x, yq);
dinvC2 = d_omega_inv_c2(x, yq);

f = (1 ./ C.^2 - 1) .* OMEGA^2 .* exp(1i * OMEGA * x);
% f = @(x) 0.0;

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
u(:, 1) = 1;%exp(1i .* y(:,1));

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
xp = Vplot * x; yp = Vplot * y;
uex = exp(1i * OMEGA * xp);

% surf(xp, yp, real(Vplot * u)); caxis([-1 1]); 
surf(xp, yp, abs(Vplot * u - uex)); 
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

function val = d_omega_inv_c2(x,y)
global OMEGA
val = -2 * (OMEGA^2 ./ c(x,y).^3) .* dcdx(x,y);
end

function D = spdiag(d)
global wq
D = spdiags(d, 0, length(wq), length(wq));
end

%function for calculating DtN * u_j
function dudt = rhs(u, c, d_omega_inv_c2)

global OMEGA
global invMA 
global ORDER

k = OMEGA ./ c;
k_sq = k.^2;

[a_pade, b_pade] = pade_coefficients(ORDER);
lambda_1_u = u;
for j = 1:length(a_pade)
%     lambda_1_u = lambda_1_u + (I + b_pade(j) .* A_k_sq) \ (a_pade(j) .* A_inv_k_sq_u);
    lambda_1_u = lambda_1_u + (spdiag(k_sq) + b_pade(j) .* invMA) \ (a_pade(j) .* (invMA * u));
end
lambda_1_u = 1i * k .* lambda_1_u; % mult by i*k \sum(...)

lambda_0_u = (spdiag(k_sq) + invMA) \ (-0.25 * d_omega_inv_c2 .* u);

dudt = lambda_1_u + lambda_0_u;

end
