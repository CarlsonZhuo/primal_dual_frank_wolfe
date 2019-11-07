function [time, perf] = SVRG(X, Y, tau, budget,mu)
% Stochastic Variance Reduced Gradient Descent for solving multinomial 
% logistic regression under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     
    
    T = 500;
    %N = 10;
    m = 100;
    [n, d] = size(X);
    h = size(Y,2);
    U = zeros(d,h); %LMO(get_grad(zeros(h,d),X,Y,1:n), tau);
    W = U;   
    L = norm(X,2);
    time = [];
    perf = [];    
    time_elapsed = 0;
         
    tic;
    for t = 1:T                        
        N = 50; %2^(1+t);
        if t > 1
            FG = get_grad(W,X,Y,1:n,mu);   
        end
        num = min(N, 20);
        step = floor(N / num);
        
        for k = 1:N            
            if k == 1 && t > 1
                G = FG;                
            else                
                sample = randsample(n,m)';
                GU = get_grad(U,X,Y,sample,mu);
                if t < 2
                    G = GU;
                else
                    GW = get_grad(W,X,Y,sample,mu);
                    G = GU - GW + FG;
                end
            end
            
            eta = 1.0/L; % 1 / sqrt(k);
            U = TraceProject(U - eta * G, tau);    
            if t<3 || mod(k,step)==1
                time_elapsed = time_elapsed + toc;
                time = [time; time_elapsed];
                perf = [perf; loss(U, X, Y, mu)];

                if time_elapsed > budget                    
                    return
                end                
                tic;

                fprintf('SVRG: t = %d, loss: %f, norm: %f done\n', k,perf(end,1),norm(svd(U),1));
            end
        end
        W = U;       

    end
end
