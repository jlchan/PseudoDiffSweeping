% Pade approximation of sqrt(1+x) & 4th order Runge Kutta - no matrices
%initialize vectors to store pade coefficients
function [a, b] = pade_coefficients(ORDER)
a = zeros(ORDER,1);
b = zeros(ORDER,1);

for o = 1:ORDER
    a(o) = 2 / (2 * ORDER + 1) * sin(o * pi / (2 * ORDER + 1))^2;
    b(o) = cos(o * pi / (2 * ORDER + 1))^2;
end
end