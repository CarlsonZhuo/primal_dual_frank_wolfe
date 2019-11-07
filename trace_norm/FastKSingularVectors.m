function [U,V] = FastKSingularVectors(G, k, tol, tau)
    if tol==0
        [U,S,V] = svds(G, k, 'largest');
    else
        [U,S,V] = svds(G, k, 'largest', 'Tolerance', tol); 
    end
    k=min(k,size(S,1));
    for i =1:k
            if S(i,i)>tau
                S(i,i)=tau;
                S(i+1:k,i+1:k)=0;
                break
            else
                tau = tau - S(i,i);
            end
    end
        if i<k
            U=U(:,1:i);
            V=V(:,1:i);
            S=S(1:i,1:i);
        end
        V = V*S;
end