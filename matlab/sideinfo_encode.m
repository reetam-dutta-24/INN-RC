function aux = sideinfo_encode(err, Ls, y, yt, xh)
% SIDEINFO_ENCODE  Codes the residual-compensation data.
%   err : message error flags (decoder bit ~= embedded bit)
%   Ls  : decoder soft output (logits) -> confidence contexts for err
%   y   : clipped stego, yt : unclipped stego -> overflow corrections
%   xh  : inverse couplings applied to y (gives the overflow contexts)
% Layout: [24: n1][AC(err)][24: n2][AC(overflow flags)][24: n3][AC(mag>1)]
%         [Elias-gamma of (mag-1) for the few magnitudes above 1]
[ctxE, ctxF, cand] = sideinfo_contexts(Ls, y, xh);
flags = yt(cand) ~= y(cand);
mags  = abs(yt(cand(flags)) - y(cand(flags)));
c1 = ac_encode(err(:), ctxE, 8);
c2 = ac_encode(flags(:), ctxF, 8);
big = mags > 1;                                    % almost all overflows are by 1 level
c3 = ac_encode(big(:), ones(numel(big), 1), 1);
aux = [dec2bits(numel(c1), 24), c1, dec2bits(numel(c2), 24), c2, ...
       dec2bits(numel(c3), 24), c3, gamma_encode(mags(big)' - 1)];
end

function b = dec2bits(n, w)
b = logical(mod(floor(n ./ 2.^(w-1:-1:0)), 2));
end
