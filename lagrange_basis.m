%trust me on this one!
function [V, Vr, r_interpolation] = lagrange_basis(N, r)
    r_interpolation = JacobiGL(0, 0, N);
    VDM = Vandermonde1D(N, r_interpolation);
    V = Vandermonde1D(N, r) / VDM;
    Vr = GradVandermonde1D(N, r) / VDM;
end

