function W = CndG(G, Q, beta, eta, tau)
% use standard FW to solve min (beta/2)|W - Q|^2 + <G, W> until the dual
% gap is at most eta.
    
%     [h,d] = size(G);
%     u = randn(h, 1);
%     u = u / norm(u);
%     v = randn(d, 1);
%     v = v / norm(v);	

    W = Q;
    while 1
        g = G + beta * (W - Q);
        
        V = LMO(g, tau);
        %[u, v] = FastSingularVectors(g, u, v);
        %V = -tau * u * v';
        
        D = W - V;
        gap = sum(dot(g, D));                            
        
        if gap <= eta
            return;
        end
        
        a = max(0, min(1, gap/(beta*norm(D,'fro')^2)));
        W = (1 - a) * W + a * V;
    end
    
end