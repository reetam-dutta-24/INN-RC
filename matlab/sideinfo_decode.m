function [err, yt] = sideinfo_decode(aux, Ls, y, net, s)
% SIDEINFO_DECODE  Inverse of sideinfo_encode. Returns the message error
% flags and the unclipped stego yt. The message errors are decoded first,
% because the overflow contexts need the corrected message.
a = abs(Ls(:));
edges = [0 0.5 1 1.5 2 3 4 6 inf];
ctxE = zeros(size(a));
for k = 1:8, ctxE(a >= edges(k) & a < edges(k+1)) = k; end
n1 = bits2dec(aux(1:24));
err = ac_decode(aux(25:24+n1), ctxE, 8);
p = 25 + n1;
sm = xor(Ls(:) > 0, err(:));
xh = icinn_invert(net, y, msg_to_map(sm, size(y), s));
[~, ctxF, cand] = sideinfo_contexts(Ls, y, xh);
n2 = bits2dec(aux(p:p+23)); p = p + 24;
flags = ac_decode(aux(p:p+n2-1), ctxF, 8); p = p + n2;
yt = y;
nf = sum(flags);
n3 = bits2dec(aux(p:p+23)); p = p + 24;
big = ac_decode(aux(p:p+n3-1), ones(nf, 1), 1); p = p + n3;
if nf > 0
    mags = ones(nf, 1);
    if any(big), mags(big) = 1 + gamma_decode(aux, p, sum(big)); end
    idx = cand(flags);
    sgn = 2*(y(idx) == 255) - 1;                  % 0 -> went below, 255 -> went above
    yt(idx) = y(idx) + sgn .* mags;
end
end

function n = bits2dec(b)
n = sum(double(b(:)') .* 2.^(numel(b)-1:-1:0));
end
