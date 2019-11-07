function V = LMO(W, tau, k, tol)
% Linera minimization oracle
%   return min<V, W> subject to trace norm of V <= tau.

%     options.tol = 0.0001;
%     [u, ~, v] = svds(W, 1, 'L', options);
%     V = -tau * u * v';
    [u, v] = FastSingularVectors(W);
    V = -tau * u * v';
end

