%ODE 'simplification' of Helmholtz factorization
%Created by Jesse Chan in Julia
%Modified to Matlab by Raven Johnson
%4 Oct 2022

global omega c0

c0 = 0.1;

%discrete values

omega = 25 * pi;     %wavelength

ppw = 20;     %points per wavelength

dt = 1 / (omega * ppw);     %time spacing

FinalTime = 1.0;     %end of domain

Nsteps = ceil(FinalTime / dt);     %total number of time steps

dt = FinalTime / Nsteps;     %recalculate time spacing

t = linspace(1e-14, FinalTime, Nsteps)';     %array of time steps


%error plotting
v_final = sweep(t,dt);

err = v_final - p(t);

t2 = linspace(0, FinalTime, Nsteps);

figure(1)
plot(t, abs(err))


% err_re = max(abs(real(err)));
% 
% err_im = max(abs(imag(err)));
% 
% figure(1)
% plot(t, real(p(t2)))
% hold on
% plot(t,real(v_final))
% title(['real part, ', num2str(err_re),' is real error'])
% legend('true solution','sweep solution')
% hold off
% 
% 
% figure(2)
% plot(t, imag(p(t2)))
% hold on
% plot(t,imag(v_final))
% title(['imaginary part, ', num2str(err_im),' is imaginary error'])
% legend('true solution','sweep solution')



%functions

%true solution p
function val = p(time)

global omega

val = exp(1i * omega * time);

end

%dp/dt
% function val = dpdt(time)
%
%     global omega
%
%     val = 1i * omega * p(time);
%
% end

%d2p/dt2
function val = d2pdt2(time)

global omega

val = -1 * omega^2 * p(time);

end

%wave speed
function val = c(time)
global c0
%val = 1 - 0.25 * exp(-25 * (time - 0.5).^2);     %exponential wave speed

val = 1 - c0 * sin(2 * pi * time);     %sinusoidal wave speed

end

%first derivative of wave speed
function val = dcdt(time)
global c0
%val = (25/2) * (time - 0.5) * exp(-25 * (time - 0.5).^2);

val = -2 * c0 * pi * cos(2 * pi * time);

end

%second derivative of wave speed
function val = d2cdt2(time)
global c0
%val = exp(-25 * (time - 0.5).^2) * (-625 * time.^2 + 625 * time - 143.75);

val = 4 * c0 * pi^2 * sin(2 * pi * time);

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


function v = sweep(time, dtime)

%Sweep using Runge Kutta 4 method

global omega

n = length(time);

v = zeros(2*n,1);


%backwards sweep with zero BC at FinalTime

dt_backwards = - 0.5 * dtime;    %cut dt_back in half to get midpts in back sweep

for i = n:-1:2

    %define mid time values that we need for RK4
    quarter_time = time(i) + 0.5 * dt_backwards;

    mid_time = time(i) + dt_backwards;


    %next midpoint value
    dv1 = rhs_backward(v(2*i), time(i)) + f(time(i));

    v1 = v(2*i) + 0.5 * dt_backwards * dv1;

    dv2 = rhs_backward(v1, quarter_time) + f(quarter_time);

    v2 = v(2*i) + 0.5 * dt_backwards * dv2;

    dv3 = rhs_backward(v2, quarter_time) + f(quarter_time);

    v3 = v(2*i) + dt_backwards * dv3;

    dv4 = rhs_backward(v3, mid_time) + f(mid_time);

    v(2*i-1) = v(2*i) + (1/6) * dt_backwards * (dv1 + 2 * dv2 + 2 * dv3 + dv4);


    %next whole value
    three_quarter_time = time(i) + 1.5 * dt_backwards;

    dv1 = rhs_backward(v(2*i-1), mid_time) + f(mid_time);

    v1 = v(2*i-1) + 0.5 * dt_backwards * dv1;

    dv2 = rhs_backward(v1, three_quarter_time) + f(three_quarter_time);

    v2 = v(2*i-1) + 0.5 * dt_backwards * dv2;

    dv3 = rhs_backward(v2, three_quarter_time) + f(three_quarter_time);

    v3 = v(2*i-1) + dt_backwards * dv3;

    dv4 = rhs_backward(v3, time(i-1)) + f(time(i-1));

    v(2*i-2) = v(2*i-1) + (1/6) * dt_backwards * (dv1 + 2 * dv2 + 2 * dv3 + dv4);

end

vb = v;

%forward sweep with force = vb

%     %grab only even indices of v (whole indices)
%     % ~errors are the same for even and odd indices (minimal change in
%         % neighbors) ~
%     w = zeros(n,1);
%
%     for k = 1:2*n
%
%         if mod(k,2) == 1
%
%             w(ceil(k/2)) = v(k);
%
%         end
%
%     end
%
%     v = w;
v = zeros(n, 1);
v(1) = p(0);

for j = 1:n-1

    mid_time = time(j) + 0.5 * dtime;

    dv1 = rhs_forward(v(j),time(j)) + vb(2*j-1);

    v1 = v(j) + 0.5 * dtime * dv1;

    dv2 = rhs_forward(v1,mid_time) + vb(2*j);

    v2 = v(j) + 0.5 * dtime * dv2;

    dv3 = rhs_forward(v2,mid_time) + vb(2*j);

    v3 = v(j) + dtime * dv3;

    dv4 = rhs_forward(v3,time(j+1)) + vb(2*j+1);

    v(j+1) = v(j) + (1/6) * dtime * (dv1 + 2 * dv2 + 2*dv3 + dv4);


end

end
