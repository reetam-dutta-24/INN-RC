function [msg, cover] = baseline_extract(stego, net, key)
% BASELINE_EXTRACT  Exact extraction for the baseline scheme.
[aux, y] = pee_extract(stego);
y = double(y); [H, W] = size(y); s = net.s;
[xr, mr] = finn_reveal(net, y / 255);
xh = min(max(round(255 * xr), 0), 255);
Ls = 8 * block_mean(mr, s);
[ctxE, ctxR] = baseline_contexts(Ls, xr);
n = sum(double(aux(1:24)) .* 2.^(23:-1:0));
d = ac_decode(aux(25:24+n), [ctxE; ctxR], 14);
err = d(1:numel(ctxE)); nz = d(numel(ctxE)+1:end);
k = sum(nz); p = 25 + n;
sg = aux(p:p+k-1); p = p + k;
mag = gamma_decode(aux, p, k);
r = zeros(H*W, 1); r(nz) = mag .* (2*double(sg(:)) - 1);
cover = uint8(xh + reshape(r, H, W));
sm = xor(Ls(:) > 0, err(:));
msg = xor(sm, logistic_keystream(numel(sm), key));
end
