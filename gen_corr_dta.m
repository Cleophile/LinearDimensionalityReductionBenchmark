function A = gen_corr_data(N,d,r,left_ratio,right_ratio)
    A = zeros(N,d);
    dim_order = randperm(d);
    idx = repelem(floor(d/r), r);
    residue = mod(d,r);
    for i=1:residue
        idx(i) = idx(i) + 1;
    end

    counter = 1;

    for count=idx
        % 正态分布均值
        mu = unifrnd(-10,10,[1,count]);
        % 正态分布方差
        max_seed_ratio = unifrnd(left_ratio,right_ratio);
        seed = unifrnd(0,5,[1,count-1]);
        max_seed = sum(seed) / (1-max_seed_ratio);
        seed(count) = max_seed;

        U = orth(sqrt(rand(count,count)));
        sig = U'*diag(seed)*U;
        sample = mvnrnd(mu,sig,N);
        for i=1:count
            dim = dim_order(counter);
            A(:,dim) = sample(:,i);
            counter = counter + 1;
        end
    end
end
