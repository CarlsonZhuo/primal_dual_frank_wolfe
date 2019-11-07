function [time, perf] = SFW(X, Y, tau, budget,mu)
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
    
    tic;    
    for k = 1:N            
        m = min(n, k^2);
                
        G = get_grad(W,X,Y,randsample(n,m)',mu);                          
        V = LMO(G,tau);        
        %[u, v] = FastSingularVectors(G,u,v);
        %V = -tau * u * v';                
        
        gamma = 2 / (k+1);
        W = (1 - gamma) * W + gamma * V;
        
        if mod(k, step) == 0
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            perf = [perf; loss(W, X, Y, mu)];
            fprintf('SFW: k = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1)); 
            
            if time_elapsed > budget                 
                return
            end
            tic;
        end
    end     
end
