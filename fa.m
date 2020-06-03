function [score, mapping] = fa(X, p)
    % 整理X，使每一个维度的X均为0
    % 得到每一列的均值，1代表列
    mu = mean(X, 1);
    X = X - repmat(mu, [size(X, 1) 1]);

    % n表示数据的维度
    [n, D] = size(X);
    epsilon = 1e-5;
    iter = 0;
    max_iter = 200;

    done = 0;

    % xn ~ N(W*zn, sigma)
    sigma = eye(D);
    mapping = rand(D, p);

    % EM执行部分，计时按一次E+M为一个单位
    while ~done
        iter = iter + 1;
        if iter>=max_iter
            done = 1;
        end

        % EM算法
        % E部分
        inv_cov = inv(mapping * mapping' + sigma);
        M = mapping' * inv_cov * X';
        SC = n * (eye(p) - mapping' * inv_cov * mapping) + M * M';

        % M部分
        mapping = (X' * M') / SC;
        sigma = (diag(diag(X' * X - mapping * M * X)) / n) + epsilon;

        % log-likelihood
        loglike = 0.5 * (log(det(inv_cov)) - sum(sum((inv_cov * X') .* X')) / n);

        % Check for convergence
        if iter ~=1 && abs(loglike - previous_loglike) < epsilon
            done = 1;
        end
        previous_loglike = loglike;
    end
    
    % Apply linear mapping
    score = X * mapping;
end
