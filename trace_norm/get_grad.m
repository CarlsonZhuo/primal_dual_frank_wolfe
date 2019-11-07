function grad = get_grad(W, X, Y, sample, mu)
% Compute the gradient of the multinomial logistic loss at W.
%   X: feature vectors
%   Y: labels
%   sample: examples used to compute the gradient
    
    m = length(sample);
    valid_X = X(sample,:);
    %y = -ones(m*c,1);
    %ind = sub2ind([m,c], 1:m, valid_Y');
    %y(ind) = 1;
    grad = 2*valid_X'*(valid_X*W-Y(sample,:))/m + mu*W;
end

