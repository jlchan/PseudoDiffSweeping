%Pade vs Taylor expansion of sqrt(1+x)

x = linspace(-1,1,40)';     %set range to calculate over
m = 7;     %set number of terms in approximation
f = @(x) sqrt(1+x.^2);     %declare f as anonymous function

%% Pade appr
a = zeros(m,1);     %initialize a and b terms
b = zeros(m,1);

pade = ones(length(x),1);     %initialize pade approximation

%Sum up the Pade approximation
for j = 1:m
    
    a(j) = 2 / (2 * m + 1) * sin(j * pi / (2 * m + 1))^2;
    
    b(j) = cos(j * pi / (2 * m + 1))^2;
    
    pade = pade + (a(j).*x.^2 ./ (1 + b(j).*x.^2));
    
end

%% Taylor appr

taylor_apr = ones(length(x),1);     %initialize taylor approximation
c = zeros(m,1);     %initialize taylor coefficients

%note for expnding around 0: f^(n) (0) = 1 for all n

%Sum up the Taylor approximation
for i = 0:m-1
    
    intermed = (1/2) - (0:i);     %intermediate values to calculate Taylor coeff's

    c(i+1) = prod(intermed) / factorial(i+1);     %Taylor coeff's
    
    taylor_apr = taylor_apr + c(i+1) .* x.^(2*(i+1));
    
end

%% Plot approximations and exact function
figure(1)
hold on
plot(x,pade)
plot(x,taylor_apr)
plot(x,f(x))

title("Approximations of sqrt(1+x^2)")
legend('Pade appr','Taylor appr','exact function')

%% Plot error between pade apprximation and exact function
figure(2)
hold on
semilogy(x,abs(pade-f(x)))
semilogy(x,abs(taylor_apr-f(x)))

title("Absolute errors of Pade appr and Taylor appr")
legend('Pade appr','Taylor appr')

