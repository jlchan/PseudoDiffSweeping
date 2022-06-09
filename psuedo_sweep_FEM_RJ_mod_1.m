%    
    %modified by Jesse Chan 2 June 2022
    %modified by Raven Shane Johnson ___

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 100*pi;     % Angular frequency

%Determine accuracy 
PPWx = 30;          % Points per wavelength in x-direction (marching)
PPWy = 10;          % Points per wavelength in y-direction (tangential)
ORDER = 2;          % Pseudo-diff order

c0 = 1;                                         % reference wavespeed
lambda = 2 * pi * c0 / OMEGA;                         % reference wavelength                 
Ny = round(PPWy * W / lambda);                      % N points in y-direction
Nx = round(PPWx * L / lambda);                      % N points in x-direction

dy = W / (Ny - 1);                                  % mesh size in y-direction
dx = L / (Nx - 1);                                  % mesh size in x-direction

c = @(x, y) c0 * (1 - 0.25 * peaks((x - W/2) * 10, (y - 0.0) * 10) / 8.1006);

fprintf('--------------------------------------------------- \n');
fprintf('Pseudo-diff Order = %i \n', ORDER);
fprintf('PPW x-axis = %.2g \n', lambda/dx);
fprintf('PPW y-axis = %.2g \n', lambda/dy);
fprintf('Nx x Ny = %i x %i \n', Nx, Ny);


%%

N = 2;
VX = linspace(-.5, .5, ceil(Ny / N)+1);
k = @(x) OMEGA / c(x, 0);
f = @(x) exp(-x);
[M, A, b, x_FE, global_to_local_ids] = compute_FE_system(N, VX, k, f);

%Defining arrays and allocating memory

Ny = length(x_FE);
x = zeros(Ny, Nx);           % x mesh
y = zeros(Ny, Nx);           % y mesh
I = speye(Ny, Ny);  % Identity matrix

for i = 1:Ny
    for j = 1:Nx
        x(i, j) = dx * (j - 1);
        y(i, j) = -W/2 + dy * (i-1); % centers at (-5, 5)        
    end
end

x_sweeping = linspace(0,1,Nx);

[x,y] = meshgrid(x_sweeping, x_FE);

%%

K = OMEGA ./ c(x,y);

invM = spdiags(1 ./ spdiags(M), 0, size(M, 1), size(M, 2));
LB = invM * A;

ndofs = length(x);
u = zeros(ndofs, Nx);
u(:, 1) = exp(1i * K(x, 0) .* x);

%%

u = zeros(Ny, Nx);           % wave field
u_energy = zeros(1, Nx);

%BOUNDARY CONDITION at x=0 -----------------
u(:,1) = exp(1i * K(:,1) .* x(:,1));
u_energy(1) = 1;

%1D Laplace operator
LB = spdiags([1 -2 1] .* ones(Ny,1), -1:1, Ny, Ny) / dy^2;    % Laplacian for variable wave speed
% Neumann BCs
LB(1,1) = -LB(2,1); LB(1,2) = LB(2,1);
LB(Ny,Ny) = -LB(Ny-1,Ny); 
LB(Ny,Ny-1) = -LB(Ny,Ny);

%SWEEPING IN THE X-DIRECTION
tic;
for j=1:Nx-1
    
    A = spdiags(1 ./ K(:,j), 0, Ny, Ny) * LB * spdiags(1./ K(:,j), 0, Ny, Ny);  
    
    DtN = sparse(Ny, Ny);
    for o = 1:ORDER
        DtN = DtN + sqrt_taylor_coeff(o-1) * A^(o-1);
    end
    DtN = spdiags(sqrt(1i * K(:,j)), 0, Ny, Ny) * DtN * spdiags(sqrt(1i * K(:,j)), 0, Ny, Ny);
    
    % Crank-Nicolson
    u(:,j+1) = (I - dx/2 * DtN) \ ((I + dx/2 * DtN) * u(:,j));
        
    u_energy(j+1) = norm(u(:,j+1)) / norm(u(:,1));
    
end
cpu_time = toc;
fprintf('Calculation CPU time = %0.2f \n', cpu_time);

% fprintf('Spectral radius = %.10e \n\n', ... 
%     abs(eigs((DtN + dx/2*C),(DtN - dx/2*C), 1, 'largestabs', ...
%     'FailureTreatment', 'keep', 'Tolerance',1e-8, ...
%     'SubspaceDimension', 40)) );



%% PLOTS -----------------------------
figure;
surf(x,y,real(u)); hold on;
colormap copper;
axis image
shading interp;
hcolor = colorbar;
caxis([-1 1]);
contour(x,y,c(x,y),10,'w','linewidth',1); hold off;
view(0,90);
xlim([0 L]);
ylim([-W W]/2);
xticks([0:W/5:L]);
yticks([-W/2:W/10:W/2]);
title('Real part of wave field');


h = gca;
h.FontSize = 10;



