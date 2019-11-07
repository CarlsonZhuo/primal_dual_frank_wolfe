function perf = loss(W, X, Y, mu)
% measure the performance of W    
% here Y should be the 0,1 matrix
    [n, d] = size(X);
    c = size(Y,2);
    %y = reshape(Y,n*c,1);
    %y = -ones(n*c,1);
    Z = X * W;
    %ind = sub2ind(size(Z), 1:size(X,1), Y');
    %y(ind) = 1;
    perf = norm(Z - Y,'fro')^2/n+ mu/2.0 * norm(W)*norm(W);
end