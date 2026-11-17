function [xr, mr] = finn_reveal(net, y01)
% FINN_REVEAL  Inverse pass with the lost latent set to zero.
y1 = haar_dwt(y01); y2 = zeros(size(y1,1), size(y1,2), net.msgCh);
for k = net.nBlocks:-1:1
    B = net.blk(k);
    sc = net.clamp * tanh(subnet3(B.rho, y1));
    y2 = (y2 - subnet3(B.eta, y1)) .* exp(-sc);
    y1 = y1 - subnet3(B.phi, y2);
end
xr = haar_idwt(y1); mr = y2;
end
