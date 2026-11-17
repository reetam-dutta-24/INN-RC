function [g, p] = gamma_decode(b, p, n)
% GAMMA_DECODE  Reads n Elias-gamma codes from bit vector b starting at p.
% Returns the values and the position just after the last code.
g = zeros(n, 1);
for i = 1:n
    N = 0;
    while ~b(p), N = N + 1; p = p + 1; end
    g(i) = sum(double(b(p:p+N)) .* 2.^(N:-1:0));
    p = p + N + 1;
end
end
