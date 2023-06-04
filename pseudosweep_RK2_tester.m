%written by Sebastian Acosta
%modified by Jesse Chan 22 Sep 2022
%modified by Raven Shane Johnson 24 May 2023

global a c0 alpha OMEGA ORDER

W = 1;              % Width of domain
L = 1;              % Length of domain
OMEGA = 20*pi;      % Angular frequency

%Determine accuracy
PPWx = 50;          % Points per wavelength in x-direction (marching)
ORDER = 2;          % Order of the Pade approximation

a = 25;
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

PPWx = 30;          % Points per wavelength in x-direction (marching)
Nx = round(PPWx * L / lambda);                      % N points in x-direction
dx = L / (Nx - 1);                                  % mesh size in x-direction

% construct space-time grids
t = linspace(0, L, Nx);
[x, y] = meshgrid(t, r);

% find L2 errors
uex = exp(1i * OMEGA * x);

u_RK2 = RK2_sweep(x,y,dx);
err_RK2 = abs(u_RK2 - uex);

w = repmat(full(diag(M)), 1, size(x, 2));
w = w / sum(w(:));

RK2_L2_error = sqrt(sum(sum(w .* err_RK2.^2)))



% %% time-stepping
% 
%     %Heun's method
%     %sweep backwards with zero "final" condition
% 
%     u = zeros(Ny, Nx);
% 
%     for j = Nx:-1:2
%         %along one value of t and all values of y
%         x1 = x(:,j);
%         f1 = f(x1,y(:,1));
%         du1 = -rhs(u(:,j), x1, y(:,1)) + f1;
%         u1 = u(:,j) -  dx * du1;
%         
%         x2 = x(:,j-1);
%         f2 = f(x2,y(:,1));
%         du2 = -rhs(u1,x2,y(:,1)) + f2;
%         u2 = u1 - dx * du2;
%         
%         %Calculate next column of u
%         u(:,j-1) = 0.5 * (u(:,j) + u2);
%         
%     end
%     
%     u_backwards = u;
%     
%     %initial condition
%     u = zeros(Ny, Nx);
%     u(:, 1) = 1;
%     
%     %sweep forward with backward sweep initial condition
%     for j = 1:Nx-1
%         
%         x1 = x(:,j);
%         du1 = rhs(u(:,j), x1, y(:,1)) + u_backwards(:,j);
%         u1 = u(:,j) +  dx * du1;
%         
%         x2 = x(:,j+1);
%         du2 = rhs(u1,x2,y(:,1)) + u_backwards(:,j+1);
%         u2 = u1 + dx * du2;
%         
%         u(:,j+1) = 0.5 * (u(:,j) + u2);
%         
%     end



% %% PLOTS -----------------------------
% 
% figure
% uex = exp(1i * OMEGA * x);
% err = abs(u - uex);
% 
% subplot(1,2,1)
% surf(x, y, real(u));
% colormap copper; axis image; shading interp; hcolor = colorbar;
% view(0,90); xlim([0 L]); ylim([-W W]/2);
% xticks(0:W/5:L); yticks(-W/2:W/10:W/2);
% 
% subplot(1,2,2)
% surf(x, y, abs(err))
% colormap copper; axis image; shading interp; hcolor = colorbar;
% view(0,90); xlim([0 L]); ylim([-W W]/2);
% xticks(0:W/5:L); yticks(-W/2:W/10:W/2);
% 
% w = repmat(full(diag(M)), 1, size(x, 2));
% w = w / sum(w(:));
% L2_error = sum(sum(w .* err.^2));
% title(['L2 error = ' num2str(L2_error)])
% 
% h = gca;
% h.FontSize = 10;

%% functions

function val = f(x,y)
global OMEGA

    val = (1 ./ c(x,y).^2 - 1) .* OMEGA^2 .* exp(1i * OMEGA * x); % forcing for manufactured solution

end

function val = c(x,y)
global c0 alpha a

    val = c0 * (1 - alpha * exp(-a * ((x-0.5).^2 + y.^2)));

end

function val = dcdx(x,y)
global c0 alpha a

    val = c0 * (alpha .* 2 * a .* (x - 0.5) .* exp(-a * ((x-0.5).^2 + y.^2)));

end

% computed using matlab symbolics
function val = dc2dx2(x,y)
global c0 alpha a 

    val = 2 * a * alpha * c0 * exp(-a * ((x - 1/2).^2 + y.^2)) - a^2 * alpha * c0 * exp(-a * ((x - 1/2).^2 + y.^2)) .* (2*x - 1).^2;

end

function val = d_inv_c2(x,y)

    val = -2 * (dcdx(x,y) ./ c(x,y).^3);

end

function val = d2_inv_c2(x,y)

    val = 6 * (dcdx(x,y).^2 - 2 * c(x,y) .* dc2dx2(x,y)) ./ c(x,y).^4;

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
function dudt = rhs(u,x,y)

global OMEGA
global invMA

    C = c(x, y);  
    %Cx = dcdx(x, y);
    dinvC2 = d_inv_c2(x, y);
    d2invC2 = d2_inv_c2(x, y);
    
    k = OMEGA ./ C;
    lambda_1_u = apply_lambda_1(k, u);
    lambda_0_u = (spdiag(k.^2) + invMA) \ (-0.25 * OMEGA^2 * dinvC2 .* u);
    
    tmp_1 = -0.25 * ((spdiag(k.^2) + invMA) * (OMEGA^2 * d2invC2 .* u) - (OMEGA^2 * dinvC2).^2 .* u);
    tmp_1 = (spdiag(k.^2) + invMA) \ ((spdiag(k.^2) + invMA) \ tmp_1); 
    tmp_2 = (spdiag(k.^2) + invMA) \ ((spdiag(k.^2) + invMA) \ ((0.25 * OMEGA^2 * dinvC2).^2 .* u)); 
    lambda_n1_u = (spdiag(k.^2) + invMA) \ apply_lambda_1(k, 0.5 * (tmp_1 + tmp_2));
    
    dudt = lambda_1_u + lambda_0_u + lambda_n1_u;

end

function u = RK2_sweep(x,y,dx)
global Ny

    %Heun's method
    %sweep backwards with zero "final" condition
    Nx = size(x,2);

    u = zeros(Ny, Nx);

    for j = Nx:-1:2
        %along one value of t and all values of y
        x1 = x(:,j);
        f1 = f(x1,y(:,1));
        du1 = -rhs(u(:,j), x1, y(:,1)) + f1;
        u1 = u(:,j) -  dx * du1;
        
        x2 = x(:,j-1);
        f2 = f(x2,y(:,1));
        du2 = -rhs(u1,x2,y(:,1)) + f2;
        u2 = u1 - dx * du2;
        
        %Calculate next column of u
        u(:,j-1) = 0.5 * (u(:,j) + u2);
        
    end
    
    u_backwards = u;
    
    %initial condition
    u = zeros(Ny, Nx);
    u(:, 1) = 1;
    
    %sweep forward with backward sweep initial condition
    for j = 1:Nx-1
        
        x1 = x(:,j);
        du1 = rhs(u(:,j), x1, y(:,1)) + u_backwards(:,j);
        u1 = u(:,j) +  dx * du1;
        
        x2 = x(:,j+1);
        du2 = rhs(u1,x2,y(:,1)) + u_backwards(:,j+1);
        u2 = u1 + dx * du2;
        
        u(:,j+1) = 0.5 * (u(:,j) + u2);
        
    end

end
