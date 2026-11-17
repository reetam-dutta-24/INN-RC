function net = finn_load(matfile)
% FINN_LOAD  Loads the float INN (baseline) exported by training/train_inn.py
S = load(matfile);
net.nBlocks = double(S.n_blocks); net.clamp = double(S.clamp);
net.msgCh = double(S.msg_ch); net.s = double(S.s);
names = {'phi', 'rho', 'eta'};
for k = 1:net.nBlocks
    for j = 1:3
        for l = 1:3
            pre = sprintf('blocks_%d_%s_c%d_', k-1, names{j}, l);
            net.blk(k).(names{j}).W{l} = S.([pre 'weight']);
            net.blk(k).(names{j}).b{l} = S.([pre 'bias']);
        end
    end
end
end
