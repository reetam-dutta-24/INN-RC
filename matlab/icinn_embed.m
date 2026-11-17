function yt = icinn_embed(net, x, M)
% ICINN_EMBED  Integer coupling forward pass.
%   x : cover (H x W, integer values 0..255)
%   M : message map (H/2 x W/2, values +1/-1)
%   yt: stego before clipping (integers, may leave 0..255 by a few levels)
x = double(x);
xa = cat(3, x(1:2:end, 1:2:end), x(2:2:end, 2:2:end));
xb = cat(3, x(1:2:end, 2:2:end), x(2:2:end, 1:2:end));
for k = 1:net.nBlocks
    if mod(k, 2) == 1
        xb = xb + round(icinn_phi(net.phi(k), xa, M));
    else
        xa = xa + round(icinn_phi(net.phi(k), xb, M));
    end
end
yt = zeros(size(x));
yt(1:2:end, 1:2:end) = xa(:, :, 1); yt(2:2:end, 2:2:end) = xa(:, :, 2);
yt(1:2:end, 2:2:end) = xb(:, :, 1); yt(2:2:end, 1:2:end) = xb(:, :, 2);
end
