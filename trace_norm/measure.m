function perf = measure(W, X, Y, testX, testY)
% measure the performance of W    
    Z = X * W';    
    % multinomial logistic loss
    perf(1) = mean(log(sum(exp(Z),2)) - Z(sub2ind(size(Z), 1:size(X,1), Y'))');    
    % test error rate        
    Z = testX * W';
    [~, pred] = max(Z,[],2); 
    
    perf(2) = nnz(pred - testY) / size(X,1);
end