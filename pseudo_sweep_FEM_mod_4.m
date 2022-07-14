%written by Sebastian Acosta    
    %modified by Jesse Chan 22 June 2022
    %modified by Raven Shane Johnson 12 July 2022

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 50*pi;      % Angular frequency

%Determine accuracy 
PPWx = 100;          % Points per wavelength in x-direction (marching)
PPWy = 4;           % Points per wavelength in y-direction (tangential)
ORDER = 2;          % Order of the Pade approximation

c0 = 1;                                             % reference wavespeed
lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength                 

Ny = round(PPWy * W / lambda);                      % N points in y-direction
Nx = round(PPWx * L / lambda);                      % N points in x-direction

dy = W / (Ny - 1);                                  % mesh size in y-direction
dx = L / (Nx - 1);                                  % mesh size in x-direction

alpha = 0.75;

c = @(x, y) c0 * (1 - alpha * exp(-100 * ((x-0.5).^2 + y.^2)));

dcdx = @(x,y) c0 * (alpha .* 200 .* (x - 0.5) .* exp(-100 * ((x-0.5).^2 + y.^2)));

d_omega_invc2_dx = @(x,y) -2 * (OMEGA^2 ./ c(x,y).^3) .* dcdx(x,y);

fprintf('--------------------------------------------------- \n');
fprintf('Pseudo-diff Order = %i \n', ORDER);
fprintf('PPW x-axis = %.2g \n', lambda/dx);
fprintf('PPW y-axis = %.2g \n', lambda/dy);
fprintf('Nx x Ny = %i x %i \n', Nx, Ny);


%%

N = 4;     %order of FE polyn?
VX = linspace(-.5, .5, ceil(Ny / N)+1);
k = @(x) 1; % dummy argument
f_zero = @(x) 0;
[M, A, ~, y_FE, global_to_local_ids] = compute_FE_system(N, VX, k, f_zero);

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

% Pade approximation of sqrt(1+x) & 4th order Runge Kutta - no matrices
%initialize vectors to store pade coefficients
a = zeros(ORDER,1);
b = zeros(ORDER,1);

for o = 1:ORDER
    a(o) = 2 / (2 * ORDER + 1) * sin(o * pi / (2 * ORDER + 1))^2;
    b(o) = cos(o * pi / (2 * ORDER + 1))^2;
end

%MATRIX FREE EXPLICIT METHOD
for j=1:Nx-1
    
    x_avg = 0.5 * (x_sweeping(j) + x_sweeping(j+1));
    delta_x = x_sweeping(j+1) - x_sweeping(j);
    k = @(y) OMEGA ./ c(x_avg, y);
    k_sq = @(y) k(y).^2;
    inv_k_sq = @(y) 1 ./ k_sq(y);
    [~, A_variable_k, ~, ~, ~] = compute_FE_system(N, VX, inv_k_sq, f_zero);
    
    A = invM * A_variable_k; 
    
    [m,n] = size(A);          

    k1 = DtN(A, k_sq, A_constant_k, d_omega_invc2_dx, Ny, x_avg, y_FE, ORDER, m, n, a, b, k, u(:,j));    
    u1 = u(:,j) + 0.5 * dx * k1;
    k2 = DtN(A, k_sq, A_constant_k, d_omega_invc2_dx, Ny, x_avg, y_FE, ORDER, m, n, a, b, k, u1);
    u2 = u(:,j) + 0.5 * dx * k2;
    k3 = DtN(A, k_sq, A_constant_k, d_omega_invc2_dx, Ny, x_avg, y_FE, ORDER, m, n, a, b, k, u2);
    u3 = u(:,j) + dx * k3; 
    k4 = DtN(A, k_sq, A_constant_k, d_omega_invc2_dx, Ny, x_avg, y_FE, ORDER, m, n, a, b, k, u3);

    %Calculate next column of u
    u(:,j+1) = u(:,j) + (dx / 6) * (k1 + 2*k2 + 2*k3 + k4);

    %print current computation step
    if mod(j, 100) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
    
end
 
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
colormap copper;
axis image
shading interp;
hcolor = colorbar;
caxis([-1 1]);
contour(X_Sweeping,Y,C_Matrix,10,'w','linewidth',1); hold off;
view(0,90);
xlim([0 L]);
ylim([-W W]/2);
xticks(0:W/5:L);
yticks(-W/2:W/10:W/2);
title('Real part of wave field Pade');


h = gca;
h.FontSize = 10;


%function for calculating DtN * u_j
function [u_next] = DtN(A_main, func1, A_k, func2, size_y, x_main, y_main, order, size_1, size_2, a_vec, b_vec, func3, u_current)

    A_u = A_main * u_current;

    lambda_0_u = ((spdiags(func1(y_main), 0, size_y, size_y) + A_k) \ (-0.25 * spdiags(func2(x_main, y_main), 0, size_y, size_y))) * u_current;
       
    lambda_1_u = u_current;

    for j = 1:order
        lambda_1_u  = lambda_1_u + (eye(size_1,size_2) + b_vec(j) .* A_main) \ (a_vec(j) .* A_u);
    end
    
    lambda_1_u = 1i * func3(y_main) .* lambda_1_u; % mult by i*k \sum(...)

    u_next = lambda_1_u + lambda_0_u;

end
