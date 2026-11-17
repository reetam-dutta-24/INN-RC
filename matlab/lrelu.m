function y = lrelu(x)
% LRELU  leaky ReLU with slope 0.2 (as in the trained networks)
y = max(x, 0.2*x);
end
