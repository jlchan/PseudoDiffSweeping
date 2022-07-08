%written by Sebastian Acosta    
    %modified by Jesse Chan 22 June 2022
    %modified by Raven Shane Johnson 22 June 2022

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

for j=1:Nx-1
    
    x_avg = 0.5 * (x_sweeping(j) + x_sweeping(j+1));
    delta_x = x_sweeping(j+1) - x_sweeping(j);
    k = @(y) OMEGA ./ c(x_avg, y);
    k_sq = @(y) k(y).^2;
    inv_k_sq = @(x) 1 ./ k_sq(x);   %why have we switched to x's?
    [~, A_variable_k, ~, ~, ~] = compute_FE_system(N, VX, inv_k_sq, f_zero);
    
    A = invM * A_variable_k; 
    
    [m,n] = size(A);
    
    % cond(A) = O(1/h^2)
    % sqrt(1 + A) = sum_i c_o * A^(o-1) 
    % sqrt(I + A) = I + sum_i (a_i * A) * (I + b_i * A)^{-1}
    %                            ^^(these terms should be well conditioned)
    
    % Pade approximation of sqrt(1+x)
   
    %is this still the function we want to apr if lambda_1 = i * sqrt(omega^s *
        %symb(1/c^2) + sigma)
    
    lambda_1 = speye(Ny, Ny);     %initialize for Pade appr    
    for o = 1:ORDER
        a = 2 / (2 * ORDER + 1) * sin(o * pi / (2 * ORDER + 1))^2;
        b = cos(o * pi / (2 * ORDER + 1))^2;
        lambda_1  = lambda_1 + (eye(m,n) + b.*A) \ (a.*A);
    end
    lambda_1 = spdiags(1i * k(y_FE), 0, Ny, Ny) * lambda_1; % mult by i*k \sum(...)

    %Where do these come from??

    % dp/dt = (lambda_1 + lambda_0) * u + O(1/OMEGA) \approx DtN * u
    %   => dp/dt = A * u
    %   ======>  abs(eig(I + dt * A)) < 1 = CFL condition.
    % lambda_1 * u = DtN \ 
    
    lambda_0 = (spdiags(k_sq(y_FE), 0, Ny, Ny) + A_constant_k) \ (-0.25 * spdiags(d_omega_invc2_dx(x_avg, y_FE), 0, Ny, Ny));
    %Why does A_constant_k come in here??

    DtN = lambda_1 + lambda_0;    
    
    %Jesse's Heun's method
    %u1 = u(:,j) + dx * DtN * u(:,j);  
    %u(:,j+1) = u(:,j) + dx * (DtN * (0.5 * (u(:,j) + u1)));

    %Retry Heun's
    %u1 = dx * DtN * u(:,j);
    %u2 = dx * DtN * (u(:,j) + 0.5 * u1);
    %u(:,j+1) = u(:,j) + u2;

    %4th order Runge Kutta attempt
    u1 = DtN * u(:,j);  
    u2 = DtN * (u(:,j) + 0.5 * dx * u1);
    u3 = DtN * (u(:,j) + 0.5 * dx * u2);
    u4 = DtN * (u(:,j) + dx * u3);
    u(:,j+1) = u(:,j) + (dx/6) * (u1 + 2*u2 + 2*u3 + u4);
   

    if mod(j, 100) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
    
end

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




