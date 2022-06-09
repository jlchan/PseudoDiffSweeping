% High order FE method on [-1,1]
%written by Jesse Chan 2 June 2022
%modified by Raven Shane Johnson 7 June 2022

function [M_global, A_global, b, x, global_to_local_ids] = compute_FE_system(N, VX, k, f)

    K = length(VX) - 1; % number of elements

    h = diff(VX); %calculate node spacing

    [rq, wq] = JacobiGL(0, 0, N); % ref quadrature points and weights
    [Vq, Vrq, r_interpolation] = lagrange_basis(N, rq); %calculate matrices with basis function inner product entries

    %mass matrix on [-1,1] reference element
    M = Vq' * diag(wq) * Vq;

    %assemble global matrices and RHS
    total_num_nodes = ceil(N * K + 1);
    A_global = sparse(total_num_nodes, total_num_nodes); %arrays of zeros to hold mass, stiffness, and RHS
    M_global = sparse(total_num_nodes, total_num_nodes);
    b = zeros(total_num_nodes, 1);
    %loop over elements
    for e = 1:K
        xq = VX(e) + (1 + rq) / 2 * h(e); %turning ref quad pts into og quad pts
        A = Vrq' * diag(wq .* k(xq)) * Vrq; %stiffness matrix on [-1,1] reference element
        starting_node_id = 1 + (e - 1) * N;
        node_ids = starting_node_id:(starting_node_id + N); %get global nodes for current element
        %sum over elements to construct matrices and RHS
        A_global(node_ids, node_ids) = A_global(node_ids, node_ids) + 2 / h(e) * A;
        M_global(node_ids, node_ids) = M_global(node_ids, node_ids) + h(e) / 2 * M;
        b(node_ids) = b(node_ids) + h(e) / 2 * Vq' * (wq .* f(xq));
    end
   
    % compute global-to-local nodes and FE nodes    
    x = zeros(total_num_nodes, 1);
    global_to_local_ids = zeros(N+1, K);
    for e = 1:K
        starting_node_id = 1 + (e - 1) * N;
        node_ids = starting_node_id:(starting_node_id + N);
        global_to_local_ids(:, e) = node_ids;
        x(node_ids) = VX(e) + (1 + r_interpolation) / 2 * h(e);
    end
    
end % function


