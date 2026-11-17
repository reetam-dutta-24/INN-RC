function off = icinn_phi(P, half, M)
% ICINN_PHI  Coupling sub-network: offsets (grey levels) for one half of the
% image, computed from the other half and the message map.
h = cat(3, half / 255, M);
h = lrelu(conv3x3(h, P.W{1}, P.b{1}));
h = lrelu(conv3x3(h, P.W{2}, P.b{2}));
off = 8 * conv3x3(h, P.W{3}, P.b{3});
end
