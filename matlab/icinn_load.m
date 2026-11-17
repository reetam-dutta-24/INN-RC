function net = icinn_load(matfile)
% ICINN_LOAD  Loads IC-INN weights exported by training/train_icinn.py
S = load(matfile);
net.nBlocks = double(S.n_blocks);
for k = 1:net.nBlocks
    pre = sprintf('phis_%d_', k-1);
    for l = 1:3
        net.phi(k).W{l} = S.([pre sprintf('c%d_weight', l)]);
        net.phi(k).b{l} = S.([pre sprintf('c%d_bias', l)]);
    end
end
for l = 1:4
    net.dec.W{l} = S.(sprintf('dec_c%d_weight', l));
    net.dec.b{l} = S.(sprintf('dec_c%d_bias', l));
end
end
