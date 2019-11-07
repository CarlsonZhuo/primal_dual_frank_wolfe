function [time, perf] = SVR_CGS(X, Y, tau, budget,mu)
% Stochastic Variance Reduced Conditional Gradient Sliding for solving 
% multinomial logistic regression under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     
    
    T = 500;    
    m = 100;
    [n, d] = size(X);
    h = size(Y,2);
    W = zeros(d,h); %LMO(get_grad(zeros(h,d),X,Y,1:n), tau);    
    U = W;

    time = [];
    perf = [];    
    time_elapsed = 0;
    
    tic;     
    cnt = 1;
    for t = 1:T        
        Q = W;
        if t > 1
            FG = get_grad(W,X,Y,1:n,mu);        
        end

        N = 50; %2^(t+1);
        num = min(N, 5);
        step = floor(N / num);
        
        for k = 1:N
            eta = 2*tau*tau/(cnt*cnt);
            beta = 3/cnt;
            gamma = 2 / (cnt+1);

            if k == 1 && t > 1 
                Q = CndG(FG, Q, beta, eta,tau);
            else                                
                sample = randsample(n,m)';
                Z = (1 - gamma) * U + gamma * Q;        
                GZ = get_grad(Z,X,Y,sample,mu);
                if t < 2
                    Q = CndG(GZ, Q, beta, eta,tau);
                else
                    GW = get_grad(W,X,Y,sample,mu);
                    Q = CndG(GZ - GW + FG, Q, beta, eta, tau);                
                end
            end
            
            U = (1 - gamma) * U + gamma * Q;                    
            cnt = cnt + 1;
                   
            if t<3 || mod(k,step)==1      
                time_elapsed = time_elapsed + toc;                
                time = [time; time_elapsed];
                perf = [perf; loss(U, X, Y, mu)];
                
                if time_elapsed > budget                     
                    return
                end
                tic;
            end

        end
        W = U; 
        fprintf('SVR-CGS: t = %d done, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1));       
    end
end
