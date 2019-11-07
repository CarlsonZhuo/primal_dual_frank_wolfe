function [time, perf] = blockFW(X, Y, s, tau, budget,mu)
% Stochastic Frank-Wolfe for solving multinomial logistic regression
% under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     
    
    N = 50000;
    %m = 100;
    [n, d] = size(X);
    h = size(Y,2);
    W = zeros(d,h); %LMO(get_grad(zeros(h,d),X,Y,1:n), tau);
    
    step = 5;
    time = [];
    perf = [];
    time_elapsed = 0;

%     u = randn(h, 1);
%     u = u / norm(u);
%     v = randn(d, 1);
%     v = v / norm(v);	
    eta = 0.1;
    L = norm(X,2);
    tic;    
    for k = 1:N            
        G = get_grad(W,X,Y,1:n,mu);                          
        tol = (1e-2)/k^2;
        G = W - 1.0/eta/L *G;
        [u, v] = FastKSingularVectors(G,s,tol,tau);
        %V = -tau * u * v';                
        
        W = (1 - eta) * W + eta * u*v';
        
        if k<10 || mod(k, step) == 0
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; loss(W, X, Y, mu)];
            fprintf('BlockFW: k = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1));             
            if time_elapsed > budget                 
                return
            end
            tic;
        end
    end     
end
