%written by Sebastian Acosta
%modified by Jesse Chan 24 Aug 2022
%modified by Raven Shane Johnson 16 Aug 2022

global c0 alpha OMEGA

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 50*pi;      % Angular frequency

%Determine accuracy
PPWx = 50;          % Points per wavelength in x-direction (marching)
PPWy = 4;           % Points per wavelength in y-direction (tangential)
ORDER = 2;          % Order of the Pade approximation

c0 = 1;                                             % reference wavespeed
alpha = 0.25;
lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength

Ny = round(PPWy * W / lambda);                      % N points in y-direction
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

global N VX
N = 4;     %order of FE polyn?
VX = linspace(-.5, .5, ceil(Ny / N)+1);
k = @(x) 1; % dummy argument
[M, A, ~, y_FE, global_to_local_ids] = compute_FE_system(N, VX, k, @(x) 0);
%Defining arrays and allocating memory
Ny = length(y_FE);
I = speye(Ny, Ny);  % Identity matrix

x_sweeping = linspace(0,1,Nx);

[x,y] = meshgrid(x_sweeping, y_FE);

%Beginning simulation

invM = spdiags(1 ./ spdiags(M, 0), 0, size(M, 1), size(M, 2));
A_constant_k = invM * A;

% IC
u = zeros(Ny, Nx);

u(:, 1) = exp(1i * k(y_FE) .* y_FE);

%%

%SWEEPING IN THE X-DIRECTION

tic;

X_Sweeping = zeros(Ny,Nx); %intro x matrix to use contour in plotting mechanism

Y = zeros(Ny,Nx); %intro y matrix to use contour in plotting mechanism

C_Matrix = c(x_sweeping,y_FE);

for i = 1:Nx
    
    Y(:,i) = y_FE;
    
end

for l=1:Ny
    
    X_Sweeping(l,:) = x_sweeping;
    
end

[a, b] = pade_coefficients(ORDER);

f = @(x) (1 ./ c(x ,y_FE).^2 - 1) .* OMEGA^2 .* exp(1i * OMEGA * x);
    
%MATRIX FREE EXPLICIT METHOD
for j=1:Nx-1
    
    x1 = x_sweeping(:,j);
    k1 = DtN(A_constant_k, x1, y_FE, a, b, u(:,j), f(x1));
    u1 = u(:,j) + 0.5 * dx * k1;
    
    x2 = x1 + 0.5 * dx;
    k2 = DtN(A_constant_k, x2, y_FE, a, b, u1, f(x2));
    u2 = u(:,j) + 0.5 * dx * k2;
    
    k3 = DtN(A_constant_k, x2, y_FE, a, b, u2, f(x2));
    u3 = u(:,j) + dx * k3;
    
    x4 = x1 + dx;
    k4 = DtN(A_constant_k, x4, y_FE, a, b, u3, f(x4));
    
    %Calculate next column of u
    u(:,j+1) = u(:,j) + (dx / 6) * (k1 + 2*k2 + 2*k3 + k4);
    
    %print current computation step
    if mod(j, 100) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
    
end

% backwards in time
% dx = -dx;
% for j=Nx:-1:2
%     
%     x1 = x_sweeping(:,j);
%     f = u(:,j); 
%     k1 = DtN(A_constant_k, x1, y_FE, a, b, u(:,j), f);
%     u1 = u(:,j) + 0.5 * dx * k1;
%     
%     x2 = x1 + 0.5 * dx;
%     k2 = DtN(A_constant_k, x2, y_FE, a, b, u1, f);
%     u2 = u(:,j) + 0.5 * dx * k2;
%     
%     k3 = DtN(A_constant_k, x2, y_FE, a, b, u2, f);
%     u3 = u(:,j) + dx * k3;
%     
%     x4 = x1 + dx;
%     k4 = DtN(A_constant_k, x4, y_FE, a, b, u3, f);
%     
%     %Calculate next column of u
%     u(:,j-1) = u(:,j) + (dx / 6) * (k1 + 2*k2 + 2*k3 + k4);
%     
%     %print current computation step
%     if mod(j, 100) == 0
%         fprintf('On step %d out of %d\n', j, Nx-1)
%     end
%     
%end

%print total computation time
cpu_time = toc;
fprintf('Calculation CPU time = %0.2f \n', cpu_time);


% fprintf('Spectral radius = %.10e \n\n', ...
%     abs(eigs((DtN + dx/2*C),(DtN - dx/2*C), 1, 'largestabs', ...
%     'FailureTreatment', 'keep', 'Tolerance',1e-8, ...
%     'SubspaceDimension', 40)) );

%% PLOTS -----------------------------

figure
surf(X_Sweeping,Y,real(u)); hold on;
% surf(X_Sweeping,Y,imag(u)); hold on;
colormap copper;
axis image
shading interp;
hcolor = colorbar;
caxis([-1 1]);
% contour(X_Sweeping,Y,C_Matrix,10,'w','linewidth',1); hold off;
view(0,90);
xlim([0 L]);
ylim([-W W]/2);
xticks(0:W/5:L);
yticks(-W/2:W/10:W/2);
% title('Real part of wave field Pade');


h = gca;
h.FontSize = 10;

figure
surf(c(X_Sweeping,Y));

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
val = c0 * (1 - alpha * exp(-100 * ((x-0.5).^2 + y.^2)));
end

function val = dcdx(x,y)
global c0 alpha
val = c0 * (alpha .* 200 .* (x - 0.5) .* exp(-100 * ((x-0.5).^2 + y.^2)));
end

function val = d_omega_inv_c2(x,y)
global OMEGA
val = -2 * (OMEGA^2 ./ c(x,y).^3) .* dcdx(x,y);
end

%function for calculating DtN * u_j
function [u_next] = DtN(A_k, x, y_FE, a_pade, b_pade, u_current, f)

global OMEGA
global N VX

k = @(y) OMEGA ./ c(x, y);
k_sq = @(y) k(y).^2;
inv_k_sq = @(y) 1 ./ k_sq(y);
[M, A_variable_k, ~, ~, ~] = compute_FE_system(N, VX, inv_k_sq, (@(x) 0));

A_u = (A_variable_k * u_current) ./ diag(M);

lambda_1_u = u_current;
for j = 1:length(a_pade)
    lambda_1_u = lambda_1_u + (speye(size(A_variable_k, 1)) + b_pade(j) .* A_variable_k) \ (a_pade(j) .* A_u);
end
lambda_1_u = 1i * k(y_FE) .* lambda_1_u; % mult by i*k \sum(...)

lambda_0_u = (spdiags(k_sq(y_FE), 0, size(A_k, 1), size(A_k, 2)) + A_k) \ (-0.25 * d_omega_inv_c2(x, y_FE) .* u_current);

u_next = lambda_1_u + lambda_0_u + f;

end
