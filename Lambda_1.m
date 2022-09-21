
%Lambda_1 operator (using pade approximation of 1/sqrt(1+x^2))

function op = Lambda_1(a,b,A_var_k,A,u,coeff)

    op = u;

    for j = 1:length(a)

        op = op + (speye(size(A_var_k, 1)) + b(j) .* A_var_k) \ (a(j) .* A);
    
    end
    
    op = 1i * coeff .* op;

end