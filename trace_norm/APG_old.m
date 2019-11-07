function [time, perf] = APG_old(X, Y, tau, budget,mu)
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
    eta = 0.1;
    L = 1.0/eta; %norm(X,2);
    tic;
    theta = 0;
    for k = 1:N
        %G = get_grad(W,X,Y,1:n);
        gamma = theta*theta*L
        theta = (mu-gamma + sqrt((gamma-mu)*(gamma-mu)+4*L*gamma))/2/L
        %eta = 1.0/L/ sqrt(k);
        %W = W - eta * G;
        U = W + (theta*gamma)/(gamma+mu*theta) * (V-W);
        G = get_grad(U,X,Y,1:n,mu);
        WP = TraceProject(U - eta * G, tau);
        V = (1-1/theta) *W+(1/theta) * WP;        
        W = WP;
        if mod(k, step) == 0
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; loss(W, X, Y,mu)];
            fprintf('APG: k = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1));
            
            if time_elapsed > budget                 
                return
            end
            tic;
        end
    end
        
end
