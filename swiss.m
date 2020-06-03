function X = swiss(n,noise)
    if ~exist('noise','val')
        noise = 0;
    end
    t = (3 * pi / 2) * (1 + 2 * rand(n, 1));
    height = 30 * rand(n, 1);
    X = [t .* cos(t) height t .* sin(t)] + noise * randn(n, 3);
end
