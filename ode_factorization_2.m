%ODE 'simplification' of Helmholtz factorization
%Created by Jesse Chan in Julia
%Modified to Matlab by Raven Johnson
%15 Sep 2022

global omega

%discrete values

omega = 10 * pi;     %wavelength

ppw = 100;     %points per wavelength

dt = 1 / (omega * ppw);     %time spacing

FinalTime = 1.0;     %end of domain

Nsteps = ceil(FinalTime / dt);     %total number of time steps

dt = FinalTime / Nsteps;     %recalculate time spacing

t = linspace(1e-14, FinalTime, Nsteps)';     %array of time steps


%error plotting
v_final = sweep(t,dt);

err = v_final - p(t);

t2 = linspace(0, FinalTime, Nsteps);

err_re = max(abs(real(err)));

err_im = max(abs(imag(err)));

figure(1)
plot(t, real(p(t2)))
hold on
plot(t,real(v_final))
title(['real part, ', num2str(err_re),' is real error'])
legend('true solution','sweep solution')
hold off


figure(2)
plot(t, imag(p(t2)))
hold on
plot(t,imag(v_final))
title(['real part, ', num2str(err_im),' is imaginary error'])
legend('true solution','sweep solution')



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

    %val = 1 - 0.25 * exp(-25 * (time - 0.5).^2);     %exponential wave speed

    val = 1 - 0.25 * sin(2 * pi * time);     %sinusoidal wave speed

end

%first derivative of wave speed
function val = dcdt(time) 

    %val = (25/2) * (time - 0.5) * exp(-25 * (time - 0.5).^2);

    val = -0.5 * pi * cos(2 * pi * time);

end

%second derivative of wave speed
function val = d2cdt2(time) 

    %val = exp(-25 * (time - 0.5).^2) * (-625 * time.^2 + 625 * time - 143.75);

    val = pi^2 * sin(2 * pi * time);

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


