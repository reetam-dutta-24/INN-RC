function L = icinn_decode(net, y, s)
% ICINN_DECODE  Message decoder. Returns one soft value (logit) per message
% bit; bit = L > 0. y is the stego (or an attacked copy of it).
y = double(y) / 255;
h = cat(3, y(1:2:end, 1:2:end), y(1:2:end, 2:2:end), y(2:2:end, 1:2:end), y(2:2:end, 2:2:end));
for l = 1:3
    h = lrelu(conv3x3(h, net.dec.W{l}, net.dec.b{l}));
end
h = conv3x3(h, net.dec.W{4}, net.dec.b{4});
% average over each s x s block (one bit per block)
[a, b] = size(h);
L = squeeze(mean(mean(reshape(h, s, a/s, s, b/s), 1), 3));
L = reshape(L, a/s, b/s);
end
