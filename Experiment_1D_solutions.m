%ODE 'simplification' of Helmholtz factorization
%RK 4 vs RK2 based on dt and omega
%Created by Jesse Chan in Julia
%Modified to Matlab by Raven Johnson
%20 Feb 2023

global omega c0

%discrete values
c0 = 0.25;

omega = 25*pi;     %wavelength

ppw = 5;     %points per wavelength

dt = 1 / (omega * ppw);     %time spacing

FinalTime = 1.0;     %end of domain

Nsteps = ceil(FinalTime / dt);     %total number of time steps

dt = FinalTime / Nsteps;     %recalculate time spacing

t = linspace(1e-14, FinalTime, Nsteps+1)';     %array of time steps


%error plotting
v_RK2_final = RK2_sweep(t,dt);

err_RK2 = v_RK2_final - p(t);

v_RK4_final = RK4_sweep(t,dt);

err_RK4 = v_RK4_final - p(t);


figure
plot(t,real(v_RK2_final),t,real(v_RK4_final),t,real(p(t)))
legend('RK2','RK4','true')

fprintf("time spacing: %f \n",dt)
fprintf("wavelength: %f \n",omega)
fprintf("RK4 error: %f \n",max(abs(err_RK4)))
fprintf("RK2 error: %f \n",max(abs(err_RK2)))
% fprintf("1 / omega: %f \n",1/omega)

% if(max(abs(err_RK4)) < 1/omega)
%      fprintf("RK4 better than 1/omega")
% end



%functions

%true solution p
function val = p(time)

global omega

    val = exp(1i * omega * time);

end

%d2p/dt2
function val = d2pdt2(time)

global omega

    val = -omega^2 * p(time);

end

%wave speed
function val = c(time)
global c0
    %val = 1 - 0.25 * exp(-25 * (time - 0.5).^2);     %exponential wave speed

    %val = 1 - c0 * sin(2 * pi * time);     %sinusoidal wave speed

    val = 1;     %constant wave speed

end

%first derivative of wave speed
function val = dcdt(time) 
global c0

    %val = (25/2) * (time - 0.5) * exp(-25 * (time - 0.5).^2);

    %val = -2 * c0 * pi * cos(2 * pi * time);

    val = 0;

end

%second derivative of wave speed
function val = d2cdt2(time) 
global c0
    %val = exp(-25 * (time - 0.5).^2) * (-625 * time.^2 + 625 * time - 143.75);

    %val = 4 * c0 * pi^2 * sin(2 * pi * time);

    val = 0;

end

%forcing function
function val = f(time)

    global omega
    
    val = d2pdt2(time) + (omega^2 ./ (c(time).^2)) .* p(time);

end

%forward sweep RHS
function val = rhs_forward(v, time)

    global omega

    lambda_1 = -1i * (omega ./ c(time));

    lambda_0 = -1 * dcdt(time) ./ (2 * c(time));

    lambda_neg_1 = (c(time) / (2 * 1i * omega)) *...
        (d2cdt2(time) / (2 * c(time)) - dcdt(time)^2 / (4 * c(time)^2));    

    val = -(lambda_1 + lambda_0 + lambda_neg_1) * v;

end


%backwards sweep RHS
function val = rhs_backward(v, time)

    val = -1 * rhs_forward(v,time);

end


function v = RK2_sweep(time, dtime)

%Sweep using Heun's method

    global omega

    n = length(time);

    v = zeros(n,1);

    f_val = f(time);

    %backwards sweep with zero BC at FinalTime
    
    dt_backwards = -dtime;

    for i = n:-1:2

        dv1 = rhs_backward(v(i),(time(i))) + f_val(i);

        v1 = v(i) + dt_backwards * dv1;

        dv2 = rhs_backward(v1,(time(i-1))) + f_val(i-1);

        v2 = v1 + dt_backwards * dv2;

        v(i-1) = 0.5 * (v(i) + v2);
    
    end

    vb = v;

    %forward sweep with force = vb

    v(1) = p(0);

    for j = 1:n-1

        dv1 = rhs_forward(v(j),(time(j))) + vb(j);

        v1 = v(j) + dtime * dv1;

        dv2 = rhs_forward(v1,(time(j+1))) + vb(j+1);

        v2 = v1 + dtime * dv2;

        v(j+1) = 0.5 * (v(j) + v2);

    end

end


function v = RK4_sweep(time, dtime)

%Sweep using Runge Kutta 4 method

global omega

n = length(time);

v = zeros(2*n,1);


%backwards sweep with zero BC at FinalTime

dt_backwards = -dtime;    

for i = n:-1:2

    dv1 = rhs_backward(v(i), time(i)) + f(time(i));
    
    v1 = v(i) + 0.5 * dt_backwards * dv1;
    dv2 = rhs_backward(v1, time(i) + 0.5 * dt_backwards) + f(time(i) + 0.5 * dt_backwards);
    
    v2 = v(i) + 0.5 * dt_backwards * dv2;
    dv3 = rhs_backward(v2, time(i) + 0.5 * dt_backwards) + f(time(i) + 0.5 * dt_backwards);
    
    v3 = v(i) + dt_backwards * dv3;    
    dv4 = rhs_backward(v3, time(i-1)) + f(time(i-1));
    
    v(i-1) = v(i) + (1/6) * dt_backwards * (dv1 + 2 * dv2 + 2 * dv3 + dv4);    
end

vb = v;

v = zeros(n, 1);
v(1) = p(0);

midpoint_interp_coeffs = [0.3125, 0.9375, -0.3125, 0.0625];

for j = 1:n-1    
    
    if j < (n - 2)
        vb_midpoint = dot(midpoint_interp_coeffs, vb(j:j+3));
    else
        % if we are close to the last timestep, we flip the interpolation around.
        vb_midpoint = dot(midpoint_interp_coeffs, vb(j:-1:j-3));
    end

    mid_time = time(j) + 0.5 * dtime;
    
    dv1 = rhs_forward(v(j), time(j)) + vb(j);    
    
    v1 = v(j) + 0.5 * dtime * dv1;
    dv2 = rhs_forward(v1, mid_time) + vb_midpoint;    
    
    v2 = v(j) + 0.5 * dtime * dv2;    
    dv3 = rhs_forward(v2, mid_time) + vb_midpoint;    
    
    v3 = v(j) + dtime * dv3;
    dv4 = rhs_forward(v3, time(j+1)) + vb(j+1);    
    
    v(j+1) = v(j) + (1/6) * dtime * (dv1 + 2 * dv2 + 2 * dv3 + dv4);
       
end

end
