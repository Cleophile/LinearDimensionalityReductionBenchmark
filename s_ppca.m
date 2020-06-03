function W = s_ppca(X, m)
    X = X';
    [d,n] = size(X);
    mu = mean(X,2);
    X = bsxfun(@minus,X,mu);

    tol = 1e-4;
    maxiter = 500;
    old_llh = -inf;
    I = eye(m);
    r = dot(X(:),X(:)); % total norm of X
    W = randn(d,m); 
    s = 1/randg;

    for iter = 2:maxiter
        M = W'*W+s*I;

        % 加速求逆
        U = chol(M);
        WX = W'*X;
        
        % likelihood
        logdetC = 2*sum(log(diag(U)))+(d-m)*log(s);
        T = U'\WX;
        trInvCS = (r-dot(T(:),T(:)))/(s*n);

        % llh 已去除常数
        llh = -n*(logdetC+trInvCS)/2;
        if abs(llh-old_llh) < tol*abs(old_llh)
            break
        end
        
        % E step
        Ez = M\WX;
        V = inv(U);
        Ezz = n*s*(V*V')+Ez*Ez';
        
        % M step
        U = chol(Ezz);                                           
        W = ((X*Ez')/U)/U';
        WR = W*U';
        s = (r-2*dot(Ez(:),WX(:))+dot(WR(:),WR(:)))/(n*d);
        old_llh = llh;
    end
end
