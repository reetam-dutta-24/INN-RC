function M = msg_to_map(bits, sz, s)
% MSG_TO_MAP  Arranges message bits on the (H/2/s x W/2/s) grid and repeats
% every bit s x s times -> +1/-1 map of size H/2 x W/2.
B = reshape(double(bits(:)), sz(1)/2/s, sz(2)/2/s);
M = kron(2*B - 1, ones(s));
end
