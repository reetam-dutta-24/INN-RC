function x = icinn_invert(net, yt, M)
% ICINN_INVERT  Exact inverse of icinn_embed for a known message map M.
ya = cat(3, yt(1:2:end, 1:2:end), yt(2:2:end, 2:2:end));
yb = cat(3, yt(1:2:end, 2:2:end), yt(2:2:end, 1:2:end));
for k = net.nBlocks:-1:1
    if mod(k, 2) == 1
        yb = yb - round(icinn_phi(net.phi(k), ya, M));
    else
        ya = ya - round(icinn_phi(net.phi(k), yb, M));
    end
end
x = zeros(size(yt));
x(1:2:end, 1:2:end) = ya(:, :, 1); x(2:2:end, 2:2:end) = ya(:, :, 2);
x(1:2:end, 2:2:end) = yb(:, :, 1); x(2:2:end, 1:2:end) = yb(:, :, 2);
end
