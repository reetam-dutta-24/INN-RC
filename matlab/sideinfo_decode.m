function [err, yt] = sideinfo_decode(aux, Ls, y)
% SIDEINFO_DECODE  Inverse of sideinfo_encode. Returns the message error
% flags and the unclipped stego yt.
[ctxE, ctxF, cand] = sideinfo_contexts(Ls, y);
n = sum(double(aux(1:24)) .* 2.^(23:-1:0));
code = aux(25:24+n);
d = ac_decode(code, [ctxE; ctxF], 18);
err = d(1:numel(ctxE));
flags = d(numel(ctxE)+1:end);
yt = y;
nf = sum(flags);
if nf > 0
    mags = gamma_decode(aux, 25 + n, nf);
    idx = cand(flags);
    sgn = 2*(y(idx) == 255) - 1;                  % 0 -> went below, 255 -> went above
    yt(idx) = y(idx) + sgn .* mags;
end
end
