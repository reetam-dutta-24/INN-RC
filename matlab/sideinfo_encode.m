function aux = sideinfo_encode(err, Ls, y, yt)
% SIDEINFO_ENCODE  Codes the residual-compensation data.
%   err : message error flags (decoder bit ~= embedded bit)
%   Ls  : decoder soft output (logits) -> confidence contexts for err
%   y   : clipped stego, yt : unclipped stego -> overflow corrections
% Layout: [24 bits: AC length][AC code][Elias-gamma overflow magnitudes]
[ctxE, ctxF, cand] = sideinfo_contexts(Ls, y);
flags = yt(cand) ~= y(cand);
mags  = abs(yt(cand(flags)) - y(cand(flags)));
code  = ac_encode([err(:); flags(:)], [ctxE; ctxF], 18);
aux   = [dec2bits(numel(code), 24), code, gamma_encode(mags')];
end

function b = dec2bits(n, w)
b = logical(mod(floor(n ./ 2.^(w-1:-1:0)), 2));
end
