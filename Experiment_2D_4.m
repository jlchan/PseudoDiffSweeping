%2D psedodifferential sweeping method with RK4
%written by Sebastian Acosta
%modified by Jesse Chan 22 Sep 2022
%modified by Raven Shane Johnson 29 May 2023

global a c0 alpha OMEGA ORDER invMA Ny

clc
clear

W = 1;              % Width of domain
L = 1;              % Length of domain

%Determine accuracy
ORDER = 2;          % Order of the Pade approximation
a = 25;
c0 = 1;            % reference wavespeed
alpha = 0.25;


%% build FE matrices

N = 4;
VX = linspace(-0.5, 0.5, 8);
[M, A, r, global_to_local_ids] = compute_FE_matrices(N, VX);

Ny = length(r); % r = one-dimensional finite element points 

I = speye(Ny, Ny);
invMA = M \ A;

%% sweep
%varying PPWx for 4 values of omega

figure     %new figure window

OMEGA = 25*pi;      % Angular frequency

for i = 5:5:50
    
    PPWx = i;          % Points per wavelength in x-direction (marching)
    lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength
    Nx = round(PPWx * L / lambda);                      % N points in x-direction
    dx = L / (Nx - 1);                                  % mesh size in x-direction
    
    % construct space-time grids
    t = linspace(0, L, Nx);
    [x, y] = meshgrid(t, r);
    
    % find L2 errors
    uex = exp(1i * OMEGA * x);
    
    u_RK4 = RK4_sweep(x,y,dx);
    err_RK4 = abs(u_RK4 - uex);
    
    w = repmat(full(diag(M)), 1, size(x, 2));
    w = w / sum(w(:));
    
    RK4_L2_error = sum(sum(w .* err_RK4.^2));

    semilogy(PPWx,RK4_L2_error,'ko','MarkerSize',10)
    hold on

end

OMEGA = 50*pi;      % Angular frequency

for i = 5:5:50

    PPWx = i;          % Points per wavelength in x-direction (marching)
    lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength
    Nx = round(PPWx * L / lambda);                      % N points in x-direction
    dx = L / (Nx - 1);                                  % mesh size in x-direction
    
    % construct space-time grids
    t = linspace(0, L, Nx);
    [x, y] = meshgrid(t, r);
    
    % find L2 errors
    uex = exp(1i * OMEGA * x);
    
    u_RK4 = RK4_sweep(x,y,dx);
    err_RK4 = abs(u_RK4 - uex);
    
    w = repmat(full(diag(M)), 1, size(x, 2));
    w = w / sum(w(:));
    
    RK4_L2_error = sum(sum(w .* err_RK4.^2));

    semilogy(PPWx,RK4_L2_error,'bo','MarkerSize',10)
    hold on

end

OMEGA = 100*pi;      % Angular frequency

for i = 5:5:50

    PPWx = i;          % Points per wavelength in x-direction (marching)
    lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength
    Nx = round(PPWx * L / lambda);                      % N points in x-direction
    dx = L / (Nx - 1);                                  % mesh size in x-direction
    
    % construct space-time grids
    t = linspace(0, L, Nx);
    [x, y] = meshgrid(t, r);
    
    % find L2 errors
    uex = exp(1i * OMEGA * x);
    
    u_RK4 = RK4_sweep(x,y,dx);
    err_RK4 = abs(u_RK4 - uex);
    
    w = repmat(full(diag(M)), 1, size(x, 2));
    w = w / sum(w(:));
    
    RK4_L2_error = sum(sum(w .* err_RK4.^2));

    semilogy(PPWx,RK4_L2_error,'ro','MarkerSize',10)
    hold on

end

OMEGA = 200*pi;      % Angular frequency

for i = 5:5:50

    PPWx = i;          % Points per wavelength in x-direction (marching)
    lambda = 2 * pi * c0 / OMEGA;                       % reference wavelength
    Nx = round(PPWx * L / lambda);                      % N points in x-direction
    dx = L / (Nx - 1);                                  % mesh size in x-direction
    
    % construct space-time grids
    t = linspace(0, L, Nx);
    [x, y] = meshgrid(t, r);
    
    % find L2 errors
    uex = exp(1i * OMEGA * x);
    
    u_RK4 = RK4_sweep(x,y,dx);
    err_RK4 = abs(u_RK4 - uex);
    
    w = repmat(full(diag(M)), 1, size(x, 2));
    w = w / sum(w(:));
    
    RK4_L2_error = sum(sum(w .* err_RK4.^2));

    semilogy(PPWx,RK4_L2_error,'co','MarkerSize',10)
    hold on

end


% plot settings
xlabel('points per wavelength','FontSize',20)
ylabel('L2 error','FontSize',20)
ax = gca;
ax.FontSize = 15;
ax.TickLength = [0.02, 0.05];
ax.LineWidth = 1.5;

test{1} = plot(nan, 'ko');
test{2} = plot(nan, 'bo');
test{3} = plot(nan, 'ro');
test{4} = plot(nan, 'co');

hold off

legend([test{:}], {'ppw = 5','ppw = 10','ppw = 15','ppw = 20'}, 'location', 'northwest', 'FontSize',20)


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
global OMEGA invMA

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

function u = RK4_sweep(x,y,dx)
global Ny
    %RK4 method
    % sweep backwards with zero "final" condition

    Nx = size(x,2);

    u = zeros(Ny, Nx);
    dx_backwards = -dx;
    
    for j = Nx:-1:2
    
        %along one value of t and all values of y
        x1 = x(:,j);
        f1 = f(x1,y(:,1));
        du1 = -rhs(u(:,j), x1, y(:,1)) + f1;
        u1 = u(:,j) +  0.5 * dx_backwards * du1;
        
        x2 = x1 + 0.5 * dx_backwards;
        f2 = f(x2,y(:,1));
        du2 = -rhs(u1, x2, y(:,1)) + f2;
        u2 = u(:,j) + 0.5 * dx_backwards * du2;
        du3 = -rhs(u2, x2, y(:,1)) + f2;
        u3 = u(:,j) + dx_backwards * du3;
    
        x3 = x(:,j-1);
        f3 = f(x3,y(:,1));
        du4 = -rhs(u3, x3, y(:,1)) + f3;
        
        %Calculate next column of u
        %keyboard
        u(:,j-1) = u(:,j) + (1/6) * dx_backwards * (du1 + 2 * du2 + 2 * du3 + du4);  
        
    end
    
    u_backwards = u;
    
    % initial condition
    u = zeros(Ny, Nx);
    u(:, 1) = 1; 
        
    % sweep forward with backward sweep initial condition
    for j = 1:Nx-1
    
        if j < Nx - 2
            u_backwards_mid = 0.3125 * u_backwards(:, j) + 0.9375 * u_backwards(:, j+1) - 0.3125 * u_backwards(:, j+2) + 0.0625 * u_backwards(:, j+3);    
        else
            % if we are close to the last timestep, we flip the interpolation around.
            u_backwards_mid = 0.3125 * u_backwards(:, j) + 0.9375 * u_backwards(:, j-1) - 0.3125 * u_backwards(:, j-2) + 0.0625 * u_backwards(:, j-3);
        end
        
        x1 = x(:,j);
        du1 = rhs(u(:,j), x1, y(:,1)) + u_backwards(:,j);
        u1 = u(:,j) +  0.5 * dx * du1;
        
        x2 = x1 + 0.5 * dx;
        du2 = rhs(u1, x2, y(:,1)) + u_backwards_mid;
        u2 = u(:,j) + 0.5 * dx * du2;
        du3 = rhs(u2, x2, y(:,1)) + u_backwards_mid;
        u3 = u(:,j) + dx * du3;
    
        x3 = x(:,j+1);
        du4 = rhs(u3, x3, y(:,1)) + u_backwards(:,j+1);
        
        %Calculate next column of u
        u(:,j+1) = u(:,j) + (1/6) * dx * (du1 + 2 * du2 + 2 * du3 + du4);      
        
    end

end
