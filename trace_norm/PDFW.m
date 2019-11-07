function [time, perf] = PDFW(X, Y,k, tau, budget,mu)
% Stochastic Frank-Wolfe for solving multinomial logistic regression
% under a trace norm constraint
    
%   Input: 
%   X = feature vectors    
%   Y = labels
%   tau = trace norm bound

%   Output:     

    %m = 100;
    
    [n, d] = size(X);
    h = size(Y,2);
    y = Y;
    N = 50000;
    W = zeros(d,h); %LMO(get_grad(zeros(h,d),X,Y,1:n), tau);
    %primal variable
    D = zeros(n,h);
    %dual variable
    XW = zeros(n,h); %=XW
    XT = X';
    V = W;
    XD = zeros(d,h); %=X'D
    step = 5;
    time = [];
    perf = [];
    L = 5;%norm(X,2);
    eta = 0.25;
    delta = 0.1; %mu/2;
    time_elapsed = 0;
    etaX = eta*X;
    ypos = 0.5-0.5*y;
    yneg = -0.5-0.5*y;
    %k= 10;
    m = min(max(ceil(k*d/n),10*k),n);
    tic;    
    for iter = 1:N            
        if iter>1
            G = XD/n + mu*W;
            tol = (1e-2)/iter;
            G = W - 1.0/eta/L *G;
        %G = -1.0/mu/n*XD;
        %if iter>10 && perf(end-1,1)<perf(end,1)
        %    [u, v] = FastKSingularVectors(G, k, tol, 0);
        %else
        [u, v] = FastKSingularVectors(G, k, tol, tau);
        %end
        WP = W*(1-eta) + eta*u*v';
        V = WP + (iter-2)/(iter+1) * (WP-W);
        %XW = X*V;
        %XW = ((1-eta)-eta*(iter-2)/(iter+1))*XW + eta*(2*iter-1)/(iter+1)*X*u*v';
        W = WP;
        
        %W = W*(1-eta) + eta*G;
        %W = TraceProject(W, tau);
        %XW = (1-eta) * XW + (etaX*u)*v';
        %W = WP;
        XW = X*V;
        end
        %D = (delta/n*(XW-y)+D)/(2*delta/n+1);
        
        %Gfstar = fstar_grad(y,D);
        %fprintf('#nonzeros %d\n',nnz(Gfstar))
        %Dp = 1./delta*D-1.0/n*(y-XW);
        %Dp = Dp/(2.0/n+1.0/delta);
        %Dp(Dp>ypos)=ypos(Dp>ypos);
        %Dp(Dp<yneg)=yneg(Dp<yneg);
        %dD=Dp-D;
        %XD =XD+XT(:,ind)*dD(ind,:);
        D = 1/delta * D-1.0/n * (y-XW);
        D = D/(2.0/n+1.0/delta);
        %D = -(y-XW)/2;
        
        %D = D - delta/n * (Gfstar - XW);
        %D(D<0 & yneg)=0;
        %D(D>1 & yneg)=1;
        %D(D>0 & ypos)=0;
        %D(D<-1 & ypos)=-1;
        %D(D>ypos) = ypos(D>ypos);
        %D(D<yneg) = yneg(D<yneg);
        %[v,ind] = maxk(sum(dD.*dD,2),m);
        %D(ind,:)=D(ind,:)+dD(ind,:);        
        %XD = XD + XT(:,ind)*dD(ind,:);        
        XD = XT*D;
        if iter<10 || mod(iter, step) == 1
            time_elapsed = time_elapsed + toc;
            time = [time; time_elapsed];
            XW = X*V;
            newperf = loss(W, X, Y, mu);
            perf = [perf; newperf];
            %perf(end,1) = min(perf(:,1));
            %perf(end,2) = min(perf(:,2));
            fprintf('PDFW: kk = %d: %f, norm: %f\n', iter, perf(end,1),norm(svd(W),1)); 
            
            if time_elapsed > budget% || iter>2*step && perf(end,1)>perf(end-1,1)-1e-6  
                L
                m
                return
            end
            tic;
        end
    end     
end



function result = fstar_grad(y,D)
    result = y+D;
    tmp = y.*D;
    result(tmp>0) = 0;
    result(tmp<-1) = 0;
end