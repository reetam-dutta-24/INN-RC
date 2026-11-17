function aux = baseline_side_encode(err, Ls, r, xr)
% BASELINE_SIDE_ENCODE  [24: AC length][AC(err, nonzero flags)][signs][gamma |r|]
[ctxE, ctxR] = baseline_contexts(Ls, xr);
nz = r(:) ~= 0;
code = ac_encode([err(:); nz], [ctxE; ctxR], 14);
aux = [logical(mod(floor(numel(code) ./ 2.^(23:-1:0)), 2)), code, (r(nz) > 0)', gamma_encode(abs(r(nz))')];
end
