function [stego, info] = baseline_embed(cover, msg, net, key)
% BASELINE_EMBED  Baseline: float INN steganography + residual compensation.
% The receiver runs the inverse INN with z = 0; whatever it gets wrong
% (cover residual r and wrong message bits) is coded and hidden with PEE.
cover = double(cover);
[H, W] = size(cover); s = net.s;
nb = (H/2/s) * (W/2/s);
msg = logical(msg(:));
sm = xor(msg, logistic_keystream(nb, key));
M = msg_to_map(sm, [H W], s) / 2;                     % +-0.5
st = finn_forward(net, cover / 255, M);
y = min(max(round(255 * st), 0), 255);
[xr, mr] = finn_reveal(net, y / 255);
xh = min(max(round(255 * xr), 0), 255);
r = cover - xh;                                       % cover residual
Ls = 8 * block_mean(mr, s);
err = xor(Ls(:) > 0, sm);
aux = baseline_side_encode(err, Ls, r, xr);
[stego, pinfo] = pee_embed(uint8(y), aux);
info.psnr_inn = 10*log10(255^2 / mean((y(:) - cover(:)).^2));
info.ber_clean = mean(err);
info.res_nonzero = mean(r(:) ~= 0);
info.aux_bits = numel(aux);
info.pee_layers = pinfo.layers;
info.payload_bits = nb;
end
