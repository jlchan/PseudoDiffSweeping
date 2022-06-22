%written by Sebastian Acosta    
    %modified by Jesse Chan 9 June 2022
    %modified by Raven Shane Johnson 21 June 2022

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 50*pi;     % Angular frequency

%Determine accuracy 
PPWx = 30;          % Points per wavelength in x-direction (marching)
PPWy = 4;          % Points per wavelength in y-direction (tangential)
ORDER = 2;          % Pseudo-diff order

c0 = 1;                                             % reference wavespeed
lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength                 

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

N = 5;
VX = linspace(-.5, .5, ceil(Ny / N)+1);
k = @(x) 1; % dummy argument
f_zero = @(x) 0;
[M, ~, ~, y_FE, global_to_local_ids] = compute_FE_system(N, VX, k, f_zero);

%Defining arrays and allocating memory
Ny = length(y_FE);
I = speye(Ny, Ny);  % Identity matrix

x_sweeping = linspace(0,1,Nx);

[x,y] = meshgrid(x_sweeping, y_FE);

%Beginning simulation

invM = spdiags(1 ./ spdiags(M, 0), 0, size(M, 1), size(M, 2));

% IC
u = zeros(Ny, Nx);

u(:, 1) = exp(1i * k(y_FE) .* y_FE);

u_tay = u;
u_pade = u;

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
    
    k = @(x) OMEGA ./ c(x_sweeping(j), x);
    inv_k_sq = @(x) 1 ./ k(x).^2;
    [~, LB, ~, ~, ~] = compute_FE_system(N, VX, inv_k_sq, f_zero);
    
    A = invM * LB; 
    
    [m,n] = size(A);
    
    % cond(A) = O(1/h^2)
    % sqrt(1 + A) = sum_i c_o * A^(o-1) 
    % sqrt(I + A) = I + sum_i (a_i * A) * (I + b_i * A)^{-1}
    %                            ^^(these terms should be well conditioned)
    
    DtN_tay = sparse(Ny, Ny);     %initialize for Taylor appr
    DtN_pade = speye(Ny, Ny);     %initialize for Pade appr
    
    %Are these orders matched up??
    
    %Taylor approximation
    for o = 1:ORDER        
        DtN_tay = DtN_tay + sqrt_taylor_coeff(o-1) * A^(o-1);
    end    
    DtN_tay = spdiags(1i * k(y_FE), 0, Ny, Ny) * DtN_tay; % mult by i*k \sum(...)
    
    %Pade approximation
    a = zeros((ORDER-1),1);
    b = zeros((ORDER-1),1);
    for o = 1:(ORDER-1)
        a(o) = 2 / (2 * (ORDER-1) + 1) * sin(o * pi / (2 * (ORDER-1) + 1))^2;
        b(o) = cos(o * pi / (2 * (ORDER-1) + 1))^2;
        DtN_pade  = DtN_pade + (eye(m,n) + b(o).*A) \ (a(o).*A);
    end
    DtN_pade = spdiags(1i * k(y_FE), 0, Ny, Ny) * DtN_pade; % mult by i*k \sum(...)
        
    % Crank-Nicolson
    u_tay(:,j+1) = (I - dx/2 * DtN_tay) \ ((I + dx/2 * DtN_tay) * u_tay(:,j));
    u_pade(:,j+1) = (I - dx/2 * DtN_pade) \ ((I + dx/2 * DtN_pade) * u_pade(:,j));            
    
    if mod(j, 100) == 0
        fprintf('On step %d out of %d\n', j, Nx-1)
    end
    
end

cpu_time = toc;
    fprintf('Calculation CPU time = %0.2f \n', cpu_time);

%!! OMEGA = 10pi, ORDER = 4: max(max(abs(u_tay-u_pade))) returns 0.1414 !!
%!! OMEGA = 50pi, ORDER = 4: max(max(abs(u_tay-u_pade))) returns 0.0680 !!
%!! OMEGA = 50pi, ORDER = 1: max(max(abs(u_tay-u_pade))) returns 2.6505 !!
%!! OMEGA = 50pi, ORDER = 2: max(max(abs(u_tay-u_pade))) returns 0.6037 !!


% fprintf('Spectral radius = %.10e \n\n', ... 
%     abs(eigs((DtN + dx/2*C),(DtN - dx/2*C), 1, 'largestabs', ...
%     'FailureTreatment', 'keep', 'Tolerance',1e-8, ...
%     'SubspaceDimension', 40)) );

%% PLOTS -----------------------------
figure(1)
surf(X_Sweeping,Y,real(u_tay)); 
hold on;
colormap copper;
axis image
shading interp;
hcolor = colorbar;
caxis([-1 1]);
contour(X_Sweeping,Y,C_Matrix,10,'w','linewidth',1); %C is maybe the proble, a function now not a matrix
hold off;
view(0,90);
xlim([0 L]);
ylim([-W W]/2);
xticks([0:W/5:L]);
yticks([-W/2:W/10:W/2]);
title('Real part of wave field Taylor');


h = gca;
h.FontSize = 10;

figure(2)
surf(X_Sweeping,Y,real(u_pade)); hold on;
colormap copper;
axis image
shading interp;
hcolor = colorbar;
caxis([-1 1]);
contour(X_Sweeping,Y,C_Matrix,10,'w','linewidth',1); hold off;
view(0,90);
xlim([0 L]);
ylim([-W W]/2);
xticks([0:W/5:L]);
yticks([-W/2:W/10:W/2]);
title('Real part of wave field Pade');


h = gca;
h.FontSize = 10;




