function [stego, info] = inncr_embed(cover, msg, net, s, key)
% INNCR_EMBED  Proposed lossless embedding (IC-INN + residual compensation).
%
%   cover : 8-bit grayscale image (H, W multiples of 2*s)
%   msg   : message bits, numel = (H/2/s)*(W/2/s)
%   net   : weights from icinn_load
%   s     : repetition factor (2, 4 or 8)
%   key   : secret key (scrambling)
%
% Steps
%   1. scramble message, build message map M
%   2. integer coupling network: yt = x + round(...)          (exactly invertible)
%   3. clip to 0..255 -> y, decode y once more with the network decoder
%   4. side information = message bits the decoder gets wrong + overflow
%      corrections, coded with the context-adaptive arithmetic coder
%      (contexts come from the decoder's own soft output and from y, so the
%      receiver can rebuild them)
%   5. hide the side information in y with multi-layer PEE -> stego

cover = double(cover);
[H, W] = size(cover);
nb = (H/2/s) * (W/2/s);
msg = logical(msg(:));
assert(numel(msg) == nb, 'message must have %d bits', nb);
ks = logistic_keystream(nb, key);
sm = xor(msg, ks);                                  % scrambled message
M  = msg_to_map(sm, [H W], s);

yt = icinn_embed(net, cover, M);
y  = min(max(yt, 0), 255);

% ---- side information ----
Ls = icinn_decode(net, y, s);
err = xor(Ls(:) > 0, sm);                           % bits the decoder gets wrong
xh = icinn_invert(net, y, M);                      % what the receiver will compute
aux = sideinfo_encode(err, Ls, y, yt, xh);

% ---- reversible embedding of the side information ----
[stego, pinfo] = pee_embed(uint8(y), aux);

info.psnr_inn = psnr8(uint8(y), uint8(cover));
info.ber_clean = mean(err);
info.n_overflow = sum(yt(:) ~= y(:));
info.aux_bits = numel(aux);
info.pee_layers = pinfo.layers;
info.payload_bits = nb;
info.bpp = nb / (H*W);
end

function p = psnr8(a, b)
mse = mean((double(a(:)) - double(b(:))).^2);
p = 10*log10(255^2 / max(mse, eps));
end
