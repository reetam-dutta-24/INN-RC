function [ctxE, ctxF, cand] = sideinfo_contexts(Ls, y)
% SIDEINFO_CONTEXTS  Contexts for the arithmetic coder. Both sides can
% compute them: they only depend on the decoder output and the clipped stego.
%   message errors : 8 confidence bins of |logit|                (1..8)
%   overflow flags : pixel value (0/255) x neighbours at the same
%                    extreme (0..4)                               (9..18)
a = abs(Ls(:));
edges = [0 0.5 1 1.5 2 3 4 6 inf];
ctxE = zeros(size(a));
for k = 1:8, ctxE(a >= edges(k) & a < edges(k+1)) = k; end
cand = find(y == 0 | y == 255);                       % only these can have overflowed
[H, W] = size(y);
yp = padarray(y, [1 1], -1);
[r, c] = ind2sub([H W], cand);
r = r + 1; c = c + 1;
v = y(cand);
same = (yp(sub2ind(size(yp), r-1, c)) == v) + (yp(sub2ind(size(yp), r+1, c)) == v) + ...
       (yp(sub2ind(size(yp), r, c-1)) == v) + (yp(sub2ind(size(yp), r, c+1)) == v);
ctxF = 8 + 1 + same + 5*(v == 255);
end
