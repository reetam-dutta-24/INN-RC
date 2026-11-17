function [msg, cover] = inncr_extract(stego, net, s, key)
% INNCR_EXTRACT  Exact extraction: returns the message bits and the
% original cover, bit for bit.
[aux, y] = pee_extract(stego);                 % side info + stego before PEE
y = double(y);
[H, W] = size(y);
Ls = icinn_decode(net, y, s);                  % same soft output as at the sender
[err, yt] = sideinfo_decode(aux, Ls, y);
sm = xor(Ls(:) > 0, err(:));                   % corrected scrambled message
msg = xor(sm, logistic_keystream(numel(sm), key));
M = msg_to_map(sm, [H W], s);
cover = uint8(icinn_invert(net, yt, M));
end
