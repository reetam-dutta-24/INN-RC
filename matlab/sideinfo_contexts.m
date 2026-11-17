function [ctxE, ctxF, cand] = sideinfo_contexts(Ls, y, xh)
% SIDEINFO_CONTEXTS  Contexts for the arithmetic coder. Both sides can
% compute them: they depend only on the decoder output Ls, the clipped
% stego y and xh = inverse couplings applied to the clipped stego.
%   message errors : 8 confidence bins of |logit|                     (1..8)
%   overflow flags : candidates are pixels at 0 or 255. The context is the
%                    offset the network applied there, estimated as
%                    d = xh - y (for 0) or y - xh (for 255), in 4 bins,
%                    times 2 (pixel value)                              (1..8)
a = abs(Ls(:));
edges = [0 0.5 1 1.5 2 3 4 6 inf];
ctxE = zeros(size(a));
for k = 1:8, ctxE(a >= edges(k) & a < edges(k+1)) = k; end
cand = find(y == 0 | y == 255);
v = y(cand);
d = xh(cand) - y(cand);
d(v == 255) = -d(v == 255);                 % positive = pushed outwards
bin = 1 + (d >= 1) + (d >= 2) + (d >= 4);   % <=0, 1, 2-3, >=4
ctxF = bin + 4*(v == 255);
end
