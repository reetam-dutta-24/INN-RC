function b = gamma_encode(g)
% GAMMA_ENCODE  Elias-gamma code of positive integers g (row vector of bits).
b = false(1, 0);
if isempty(g), return; end
g = double(g(:));
N = floor(log2(g));
len = 2*N + 1;
b = false(1, sum(len));
p = 0;
for i = 1:numel(g)
    p = p + N(i);                                  % N(i) leading zeros
    b(p+1:p+N(i)+1) = logical(mod(floor(g(i) ./ 2.^(N(i):-1:0)), 2));
    p = p + N(i) + 1;
end
end
