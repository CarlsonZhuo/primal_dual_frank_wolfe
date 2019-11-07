function [time, perf] = SVR_FW(X, Y, tau, budget,mu)
% Stochastic Variance Reduced Frank-Wolfe for solving multinomial logistic 
% regression under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     
    
    T = 500;
    %N = 10;
    %m = 1;
    [n, d] = size(X);
    h = size(Y,2);
    U = zeros(d,h); %LMO(get_grad(zeros(h,d),X,Y,1:n), tau);
    W = U;   
    
    time = [];
    perf = [];
    time_elapsed = 0;
    
%     u = randn(h, 1);
%     u = u / norm(u);
%     v = randn(d, 1);
%     v = v / norm(v);	
    
    cnt = 1;
    tic;         
    for t = 1:T                               
        N = 50; %2^(3+t);
        if t > 1
            FG = get_grad(W,X,Y,1:n,mu);
        end

        num = min(N, 10);
        step = floor(N / num);
   
        for k = 1:N      
            m = min(cnt,n);
            if k == 1 && t > 1
                V = LMO(FG,tau);                
                %[u, v] = FastSingularVectors(FG, u, v);
                %V = -tau * u * v';
            else                    
                sample = randsample(n,m)';
                GU = get_grad(U,X,Y,sample,mu);
                if t < 2
                    V = LMO(GU,tau);
                else
                    GW = get_grad(W,X,Y,sample,mu);
                    V = LMO(GU - GW + FG,tau);
                end
                %[u, v] = FastSingularVectors(GU-GW+FG, u, v);
                %V = -tau * u * v';
            end
            
            gamma = 2 / (cnt+1);
            U = (1 - gamma) * U + gamma * V;    
            
          
            if t<3 || mod(k,step)==1      
                time_elapsed = time_elapsed + toc;
                time = [time; time_elapsed];
                perf = [perf; loss(U, X, Y, mu)];
                
                if time_elapsed > budget                     
                    return
                end                
                tic;
            end
            
            cnt = cnt + 1;
        end
        W = U;   
        fprintf('SVR-FW: t = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(W),1));
    end
end
