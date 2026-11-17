function run_ablation_aux(s)
% RUN_ABLATION_AUX  Size of the side information under four coding choices
% (all test images, quality model, payload s):
%   A0  no entropy coding: raw error flags + raw overflow flags + gamma magnitudes
%   A1  arithmetic coding, one context per stream
%   A2  arithmetic coding, confidence contexts for errors and
%       neighbourhood contexts for overflow (our first version)
%   A3  proposed: confidence contexts + predicted-offset contexts + AC magnitude flag
if nargin < 1, s = 4; end
if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
root = fileparts(fileparts(mfilename('fullpath')));
net = icinn_load(fullfile(root, 'models', 'icinn.mat'));
sets = {'test_bsd100', 'test_kodak', 'test_medical'};
fid = fopen(fullfile(root, 'results', sprintf('ablation_aux_s%d.csv', s)), 'w');
fprintf(fid, 'set,image,bits,n_err,n_cand,n_ovf,A0,A1,A2,A3\n');
for d = 1:numel(sets)
    files = dir(fullfile(root, 'data', sets{d}, '*.png'));
    for i = 1:numel(files)
        X = double(imread(fullfile(root, 'data', sets{d}, files(i).name)));
        nb = numel(X) / 4 / s^2;
        rand('seed', 1000*d + i + s); m = rand(nb, 1) > 0.5;   %#ok<RAND>
        M = msg_to_map(m, size(X), s);
        yt = icinn_embed(net, X, M); y = min(max(yt, 0), 255);
        Ls = icinn_decode(net, y, s); err = xor(Ls(:) > 0, m);
        xh = icinn_invert(net, y, M);
        [ctxE, ctxF, cand] = sideinfo_contexts(Ls, y, xh);
        flags = yt(cand) ~= y(cand); mags = abs(yt(cand(flags)) - y(cand(flags)));
        g = numel(gamma_encode(mags'));
        A0 = numel(err) + numel(flags) + g;
        A1 = numel(ac_encode(err, ones(size(err)), 1)) + numel(ac_encode(flags, ones(size(flags)), 1)) + g;
        % neighbourhood contexts (first version)
        [H, W] = size(y); yp = -ones(H+2, W+2); yp(2:end-1, 2:end-1) = y;
        [r, c] = ind2sub([H W], cand); r = r + 1; c = c + 1; v = y(cand);
        same = (yp(sub2ind(size(yp), r-1, c)) == v) + (yp(sub2ind(size(yp), r+1, c)) == v) + ...
               (yp(sub2ind(size(yp), r, c-1)) == v) + (yp(sub2ind(size(yp), r, c+1)) == v);
        A2 = numel(ac_encode(err, ctxE, 8)) + numel(ac_encode(flags, 1 + same + 5*(v == 255), 10)) + g;
        A3 = numel(sideinfo_encode(err, Ls, y, yt, xh)) - 72;      % without the three length fields
        fprintf(fid, '%s,%s,%d,%d,%d,%d,%d,%d,%d,%d\n', sets{d}, files(i).name, nb, sum(err), numel(cand), sum(flags), A0, A1, A2, A3);
    end
    fprintf('%s done\n', sets{d});
end
fclose(fid);
end
