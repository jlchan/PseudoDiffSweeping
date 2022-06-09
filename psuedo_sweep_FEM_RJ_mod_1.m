%    
    %modified by Jesse Chan 2 June 2022
    %modified by Raven Shane Johnson ___

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 100*pi;     % Angular frequency

%Determine accuracy 
PPWx = 20;          % Points per wavelength in x-direction (marching)
PPWy = 10;          % Points per wavelength in y-direction (tangential)
ORDER = 2;          % Pseudo-diff order

c0 = 1;                                               % reference wavespeed
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

N = 1;
VX = linspace(-.5, .5, ceil(Ny / N)+1);
k = @(x) OMEGA ./ c(x, 0);
f = @(x) exp(-x);
[M, ~, ~, y_FE, global_to_local_ids] = compute_FE_system(N, VX, k, f);

%Defining arrays and allocating memory
Ny = length(y_FE);
I = speye(Ny, Ny);  % Identity matrix

x_sweeping = linspace(0,1,Nx);

[x,y] = meshgrid(x_sweeping, y_FE);

%Beginning simulation

K = @(x,y) OMEGA ./ c(x,y);

invM = spdiags(1 ./ spdiags(M, 0), 0, size(M, 1), size(M, 2));

% IC
u = zeros(Ny, Nx);
u(:, 1) = exp(1i * K(0, y_FE) .* x(:, 1));


%%

%SWEEPING IN THE X-DIRECTION
tic;
for j=1:Nx-1
    
    k = @(x) OMEGA ./ c(x_sweeping(j), x);    
    f = @(x) 0;
    [M, LB, b, y_FE, global_to_local_ids] = compute_FE_system(N, VX, k, f);
    
    LB = invM * LB;     %overwrite A
    A = LB;
    
    % sqrt(1 + LB_operator) = Taylor series
    DtN = sparse(Ny, Ny);    
    for o = 1:ORDER        
        DtN = DtN + sqrt_taylor_coeff(o-1) * A^(o-1);   
    end

%     k_j = k(y_FE); 
%     A = spdiags(1 ./ k_j, 0, Ny, Ny) * LB * spdiags(1./ k_j, 0, Ny, Ny); 
%     DtN = sparse(Ny, Ny);
%     for o = 1:ORDER
%         DtN = DtN + sqrt_taylor_coeff(o-1) * A^(o-1);
%     end
%     DtN = spdiags(sqrt(1i * k_j), 0, Ny, Ny) * DtN * spdiags(sqrt(1i * k_j), 0, Ny, Ny);
        
    % Crank-Nicolson
    u(:,j+1) = (I - dx/2 * DtN) \ ((I + dx/2 * DtN) * u(:,j));            
    
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
title('Real part of wave field');

h = gca;
h.FontSize = 10;



