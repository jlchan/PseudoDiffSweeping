

function a = sqrt_taylor_coeff(n)
    if n==0
        a = 1;
    else
        a = 1/2;
        for j=1:n-1
            a = a*(2*j-1)/2;
        end
        a = a*(-1)^(n-1) / factorial(n);
    end
end

