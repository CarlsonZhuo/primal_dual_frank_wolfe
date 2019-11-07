function [p] = SimplexProject(x,tau)
% find the closest point in the simplex (scaled by tau) to x

if (nargin < 2)
    tau = 1;
end
x = x ./ tau;
    

n = length(x);
y = sortrows(x,-1);
index = zeros(n,1);

for i = 1:n
    z = y - y(i); 
    index(i) = sum( z(1:i));
end

t = -1;
for i = 2:n
    if (index(i-1) <= 1) && (index(i) >= 1) 
        t = (sum(y(1:i-1)) - 1) / (i-1);
    end
end
if (t == -1)
    t = (sum(y(1:n)) - 1) / n;
end

p = x - t;
p(p < 0) = 0;
p = p .* tau;

return;
