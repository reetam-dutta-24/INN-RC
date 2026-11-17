function run_benchmark(maxPerSet)
% RUN_BENCHMARK  Lossless benchmark on all test images.
%   Methods : PEE   - rhombus PEE embedding the message directly (classical)
%             BASE  - float INN + residual compensation (baseline, s = 4 model)
%             PROP  - IC-INN + residual compensation (proposed, quality model)
%             PROPR - same scheme, robustness-oriented model (icinn_r.mat)
%   Payload : s = 8, 4, 2  ->  256, 1024, 4096 bits on a 256x256 image
%   Output  : ../results/benchmark.csv (one row per image / method / payload)
if nargin < 1, maxPerSet = inf; end
if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
root = fileparts(fileparts(mfilename('fullpath')));
sets = {'test_bsd100', 'test_kodak', 'test_medical'};
key = 20261127;
prop = icinn_load(fullfile(root, 'models', 'icinn.mat'));
propR = icinn_load(fullfile(root, 'models', 'icinn_r.mat'));
baseFile = fullfile(root, 'models', 'finn_s4.mat');
haveBase = exist(baseFile, 'file') == 2;
if haveBase, base = finn_load(baseFile); end
if ~exist(fullfile(root, 'results'), 'dir'), mkdir(fullfile(root, 'results')); end
fid = fopen(fullfile(root, 'results', 'benchmark.csv'), 'w');
fprintf(fid, 'set,image,method,s,bits,bpp,psnr,ssim,aux_bits,layers,exact_cover,ber,nc,psnr_inn,ber_inn,t_embed,t_extract,ok\n');
for d = 1:numel(sets)
    files = dir(fullfile(root, 'data', sets{d}, '*.png'));
    for i = 1:min(numel(files), maxPerSet)
        I = imread(fullfile(root, 'data', sets{d}, files(i).name));
        for s = [8 4 2]
            nb = (size(I,1)/2/s) * (size(I,2)/2/s);
            rand('seed', 1000*d + i + s); msg = rand(nb, 1) > 0.5;   %#ok<RAND>
            methods = {'PEE', 'PROP', 'PROPR'};
            if haveBase && s == base.s, methods{end+1} = 'BASE'; end %#ok<AGROW>
            for m = 1:numel(methods)
                r = struct('psnr', NaN, 'ssim', NaN, 'aux', NaN, 'layers', NaN, 'exact', 0, ...
                           'ber', NaN, 'nc', NaN, 'pinn', NaN, 'binn', NaN, 'te', NaN, 'tx', NaN, 'ok', 0);
                try
                    switch methods{m}
                        case 'PEE'
                            tic; [S, inf] = pee_embed(I, msg); r.te = toc;
                            tic; [m2, I2] = pee_extract(S); r.tx = toc;
                            r.aux = 0; r.layers = inf.layers;
                        case 'PROP'
                            tic; [S, inf] = inncr_embed(I, msg, prop, s, key); r.te = toc;
                            tic; [m2, I2] = inncr_extract(S, prop, s, key); r.tx = toc;
                            r.aux = inf.aux_bits; r.layers = inf.pee_layers;
                            r.pinn = inf.psnr_inn; r.binn = inf.ber_clean;
                        case 'PROPR'
                            tic; [S, inf] = inncr_embed(I, msg, propR, s, key); r.te = toc;
                            tic; [m2, I2] = inncr_extract(S, propR, s, key); r.tx = toc;
                            r.aux = inf.aux_bits; r.layers = inf.pee_layers;
                            r.pinn = inf.psnr_inn; r.binn = inf.ber_clean;
                        case 'BASE'
                            tic; [S, inf] = baseline_embed(I, msg, base, key); r.te = toc;
                            tic; [m2, I2] = baseline_extract(S, base, key); r.tx = toc;
                            r.aux = inf.aux_bits; r.layers = inf.pee_layers;
                            r.pinn = inf.psnr_inn; r.binn = inf.ber_clean;
                    end
                    r.psnr = metric_psnr(S, I); r.ssim = metric_ssim(S, I);
                    r.exact = isequal(I2, I);
                    [r.ber, r.nc] = metric_ber_nc(msg, m2);
                    r.ok = 1;
                catch err
                    fprintf('  %s %s s=%d failed: %s\n', files(i).name, methods{m}, s, err.message);
                end
                fprintf(fid, '%s,%s,%s,%d,%d,%.6f,%.4f,%.6f,%g,%g,%d,%.6f,%.6f,%.4f,%.6f,%.3f,%.3f,%d\n', ...
                    sets{d}, files(i).name, methods{m}, s, nb, nb/numel(I), r.psnr, r.ssim, r.aux, r.layers, ...
                    r.exact, r.ber, r.nc, r.pinn, r.binn, r.te, r.tx, r.ok);
            end
        end
        fprintf('%s %s done\n', sets{d}, files(i).name);
    end
end
fclose(fid);
end
