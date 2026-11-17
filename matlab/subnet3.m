function y = subnet3(P, x)
% SUBNET3  conv-lrelu-conv-lrelu-conv (sub-network of the float INN blocks)
y = lrelu(conv3x3(x, P.W{1}, P.b{1}));
y = lrelu(conv3x3(y, P.W{2}, P.b{2}));
y = conv3x3(y, P.W{3}, P.b{3});
end
