function [ctxE, ctxR] = baseline_contexts(Ls, xr)
% BASELINE_CONTEXTS  message-error contexts (8 confidence bins) and residual
% contexts (6 bins of the distance of 255*xr to the rounding boundary)
a = abs(Ls(:));
edges = [0 0.5 1 1.5 2 3 4 6 inf];
ctxE = zeros(size(a));
for k = 1:8, ctxE(a >= edges(k) & a < edges(k+1)) = k; end
f = abs(255 * xr(:) - round(255 * xr(:)));           % 0 (safe) .. 0.5 (on the edge)
ctxR = 8 + min(floor(f / 0.5 * 6), 5) + 1;
end
