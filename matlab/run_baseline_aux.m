function run_baseline_aux()
% RUN_BASELINE_AUX  Side information the float-INN baseline would need
% (computed before the PEE stage, so it is known even when PEE cannot hold it).
if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
root = fileparts(fileparts(mfilename('fullpath')));
base = finn_load(fullfile(root, 'models', 'finn_s4.mat'));
sets = {'test_bsd100', 'test_kodak', 'test_medical'}; key = 20261127; s = base.s;
fid = fopen(fullfile(root, 'results', 'baseline_aux.csv'), 'w');
fprintf(fid, 'set,image,bits,psnr_inn,ber_inn,res_nonzero,aux_bits,pee_capacity\n');
for d = 1:numel(sets)
    files = dir(fullfile(root, 'data', sets{d}, '*.png'));
    for i = 1:numel(files)
        X = double(imread(fullfile(root, 'data', sets{d}, files(i).name)));
        nb = numel(X) / 4 / s^2;
        rand('seed', 1000*d + i + s); msg = rand(nb, 1) > 0.5;   %#ok<RAND>
        sm = xor(msg, logistic_keystream(nb, key));
        st = finn_forward(base, X / 255, msg_to_map(sm, size(X), s) / 2);
        y = min(max(round(255 * st), 0), 255);
        [xr, mr] = finn_reveal(base, y / 255);
        r = X - min(max(round(255 * xr), 0), 255);
        Ls = 8 * block_mean(mr, s); err = xor(Ls(:) > 0, sm);
        aux = baseline_side_encode(err, Ls, r, xr);
        cap = 0;                                    % single-layer PEE capacity of y (both passes)
        for setId = [0 1]
            [cand, pred] = pee_setup(y, setId); e = y(cand) - pred; cap = cap + sum(e == 0 | e == -1);
        end
        fprintf(fid, '%s,%s,%d,%.4f,%.6f,%.6f,%d,%d\n', sets{d}, files(i).name, nb, ...
            10*log10(255^2 / mean((y(:) - X(:)).^2)), mean(err), mean(r(:) ~= 0), numel(aux), cap);
    end
end
fclose(fid);
end
