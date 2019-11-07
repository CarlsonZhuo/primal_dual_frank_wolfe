function [time, perf] = SCGS(X, Y, tau, budget,mu)
% Stochastic Conditional Gradient Sliding for solving multinomial logistic 
% regression under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     
    
    N = 30000;
    %m = 10;
    [n, d] = size(X);
    h = size(Y,2);
    W = zeros(d,h); %LMO(get_grad(zeros(h,d),X,Y,1:n), tau);    
    Q = W;
    L =norm(X,2);
    step = 5;
    time = [];
    perf = [];
    time_elapsed = 0;
    
    %time(1) = 0;
    %perf(1,:) = measure(W, X, Y, testX, testY);
    
    tic;
    for k = 1:N        
        gamma = 2 / (k+1);
        m = 100; %min(n,k^3);
        
        Z = (1 - gamma) * W + gamma * Q;        
        G = get_grad(Z,X,Y,randsample(n,m)',mu);
        Q = CndG(G, Q, 3/k, 2*tau*tau/(k*k),tau);        
        W = (1 - gamma) * W + gamma * Q;  
        
        if mod(k, step) == 0
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; loss(W, X, Y,mu)];
            fprintf('SCGS: k = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1));    
            
            if time_elapsed > budget                 
                return
            end
            tic;
        end
    end
end
