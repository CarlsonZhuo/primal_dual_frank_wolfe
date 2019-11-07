function [time, perf] = SGD(X, Y, tau, budget,mu)
% Stochastic Gradient Descent for solving multinomial logistic regression
% under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     
    
    N = 25000000;
    m = 100;
    [n, d] = size(X);
    h = size(Y,2);
    W = zeros(d,h);
    
    step = 5;
    time = [];
    perf = [];
    time_elapsed = 0;
    L = norm(X,2);
    tic;
    for k = 1:N
        G = get_grad(W,X,Y,randsample(n,m)',mu);
        %G = get_grad(W,X,Y,1:n);
        
        eta = 1.0/L/ sqrt(k);
        %W = W - eta * G;
        W = TraceProject(W - eta * G, tau);                
        
        if mod(k, step) == 0
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; loss(W, X, Y,mu)];
            fprintf('SGD: k = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1));
            
            if time_elapsed > budget                 
                return
            end
            tic;
        end
    end
        
end
