function L = block_mean(h, s)
% BLOCK_MEAN  mean over non-overlapping s x s blocks
[a, b] = size(h);
L = reshape(mean(mean(reshape(h, s, a/s, s, b/s), 1), 3), a/s, b/s);
end
