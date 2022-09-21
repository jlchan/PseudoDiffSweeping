
%Lambda_0 operator

function op = Lambda_0(coeff_1, A, coeff_2,u)


    op = (spdiags(coeff_1, 0, size(A, 1), size(A, 2)) + A) \ (-0.25 * coeff_2 .* u);

end