function [stego, z] = finn_forward(net, x01, M)
% FINN_FORWARD  Float INN embedding (HiNet-style affine coupling in the
% Haar domain). x01: cover in [0,1], M: message map (+-0.5), H/2 x W/2.
x1 = haar_dwt(x01); x2 = M;
for k = 1:net.nBlocks
    B = net.blk(k);
    x1 = x1 + subnet3(B.phi, x2);
    sc = net.clamp * tanh(subnet3(B.rho, x1));
    x2 = x2 .* exp(sc) + subnet3(B.eta, x1);
end
stego = haar_idwt(x1); z = x2;
end
