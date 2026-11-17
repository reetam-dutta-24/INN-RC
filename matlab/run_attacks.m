function run_attacks(nBsd)
% RUN_ATTACKS  Robustness of the message channel when the stego is NOT
% delivered losslessly (noise / JPEG). Reversibility is lost in that case,
% the question is only how many message bits still come out right.
%   Images : all Kodak + all DICOM + first nBsd BSD100 images, payload s = 4
%   Output : ../results/attacks.csv
if nargin < 1, nBsd = 18; end
if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
root = fileparts(fileparts(mfilename('fullpath')));
key = 20261127; s = 4;
prop = icinn_load(fullfile(root, 'models', 'icinn.mat'));
propR = icinn_load(fullfile(root, 'models', 'icinn_r.mat'));
base = finn_load(fullfile(root, 'models', 'finn_s4.mat'));
attacks = {'none', 0; 'gauss', 1; 'gauss', 2; 'gauss', 3; 'sp', 0.005; 'sp', 0.01; 'jpeg', 95; 'jpeg', 90; 'jpeg', 80};
list = {};
f = dir(fullfile(root, 'data', 'test_kodak', '*.png'));   for i = 1:numel(f), list{end+1} = fullfile('test_kodak', f(i).name); end %#ok<AGROW>
f = dir(fullfile(root, 'data', 'test_medical', '*.png')); for i = 1:numel(f), list{end+1} = fullfile('test_medical', f(i).name); end %#ok<AGROW>
f = dir(fullfile(root, 'data', 'test_bsd100', '*.png'));  for i = 1:min(nBsd, numel(f)), list{end+1} = fullfile('test_bsd100', f(i).name); end %#ok<AGROW>
fid = fopen(fullfile(root, 'results', 'attacks.csv'), 'w');
fprintf(fid, 'image,method,attack,param,ber,nc,psnr_attacked\n');
for i = 1:numel(list)
    I = imread(fullfile(root, 'data', list{i}));
    nb = (size(I,1)/2/s) * (size(I,2)/2/s);
    rand('seed', 77 + i); msg = rand(nb, 1) > 0.5;   %#ok<RAND>
    ks = logistic_keystream(nb, key);
    S = struct();
    try, S.PROP  = inncr_embed(I, msg, prop, s, key);  catch, end
    try, S.PROPR = inncr_embed(I, msg, propR, s, key); catch, end
    try, S.BASE  = baseline_embed(I, msg, base, key);  catch, end   % fails when its side info does not fit
    try, S.PEE   = pee_embed(I, msg);                  catch, end
    meth = fieldnames(S);
    for a = 1:size(attacks, 1)
        for m = 1:numel(meth)
            randn('seed', i*100 + a); rand('seed', i*100 + a);   %#ok<RAND>
            A = apply_attack(S.(meth{m}), attacks{a,1}, attacks{a,2});
            switch meth{m}
                case 'PROP'
                    L = icinn_decode(prop, A, s); bits = xor(L(:) > 0, ks);
                case 'PROPR'
                    L = icinn_decode(propR, A, s); bits = xor(L(:) > 0, ks);
                case 'BASE'
                    [~, mr] = finn_reveal(base, double(A) / 255);
                    L = block_mean(mr, s); bits = xor(L(:) > 0, ks);
                case 'PEE'
                    try
                        bits = pee_extract(A); bits = bits(:);
                        if numel(bits) < nb, bits(end+1:nb) = rand(nb - numel(bits), 1) > 0.5; end %#ok<RAND>
                        bits = bits(1:nb);
                    catch
                        bits = rand(nb, 1) > 0.5;   %#ok<RAND>  % extraction breaks down -> guessing
                    end
            end
            [ber, nc] = metric_ber_nc(msg, bits);
            fprintf(fid, '%s,%s,%s,%g,%.6f,%.6f,%.3f\n', list{i}, meth{m}, attacks{a,1}, attacks{a,2}, ber, nc, metric_psnr(A, S.(meth{m})));
        end
    end
    fprintf('%s done\n', list{i});
end
fclose(fid);
end
