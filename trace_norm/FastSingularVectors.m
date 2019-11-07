function [u, v] = FastSingularVectors(A, u0, v0, epsilon)
	[m, n] = size(A);
	if (nargin < 2)
		u0 = randn(m, 1);
		u0 = u0 / norm(u0);
		v0 = randn(n, 1);
		v0 = v0 / norm(v0);	
		epsilon = 0.0001;
	elseif (nargin < 4)
		epsilon = 0.0001;
	end

	if (m <= n)
		B = A;
		u = u0;
	else
		B = A';
		u = v0;
	end

	w = zeros(min(m, n), 1);        
	while norm(u - w) > epsilon
		w = u;
		v = B * (B' * u);
		u = v / norm(v);                
	end
    
	v = B' * u;
	v = v / norm(v);

	if (m > n)
		w = u;
		u = v;
		v = w;
	end
end