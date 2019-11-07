% project a matrix according to the Euclidean distance onto the set of
% tracenorm <= tau
function [A] = TraceProject(M,tau)
% Input: 
% M = input m*n matrix 
% Output: 
% A = projected matrix 

[U,S,V] = svd(full(M), 'econ');

d = diag(S);

% we project onto simplex union zero, so only need simplex project if the
% sum is larger than tau. 
if (sum(d) > tau) 
    newd = SimplexProject(d,tau);
	A = U * diag(newd) * V';
else
	A = M;
end
return;
