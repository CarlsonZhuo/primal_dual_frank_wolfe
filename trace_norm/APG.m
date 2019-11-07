function [time, perf] = APG(X, Y, tau, budget,mu)
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
    U = zeros(d,h);
    V = zeros(d,h);
    
    step = 5;
    time = [];
    perf = [];
    time_elapsed = 0;
    eta = 0.001;
    %L = 1.0/eta; %norm(X,2);
    tic;
    WP = W;
    for k = 1:N
        V = WP + (k-2)/(k+1) * (WP-W);
        W = WP;
        G = get_grad(V,X,Y,1:n,mu);
        WP = TraceProject(V - eta * G, tau);
        if mod(k, step) == 1
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; loss(W, X, Y,mu)];
            fprintf('APG: k = %d, loss: %f, norm: %f done\n', k,perf(end),norm(svd(W),1));
            
            if time_elapsed > budget                 
                return
            end
            tic;
        end
    end
        
end
