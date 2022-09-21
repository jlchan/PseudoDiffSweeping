
%Lambda_-1 operator 

function op = Lambda_Neg_1(u)

    q = d_omega_inv_c2(x, y_FE) .* u;

    numerator_1 = -0.25 * (k_sq(y_FE) .* q + A_constant_k * q);
    term_1 = % invert the (spdiags(k_sq) + A_constant_k) \ numerator_1
    
    term_2 = 
    
    term_1 = Lambda_1(numerator_1, ...)

end