function make_visuals()
% MAKE_VISUALS  Saves cover / stego / difference / recovered images of a
% few test images (payload s = 4) for the figures in the paper.
if exist('OCTAVE_VERSION', 'builtin'), pkg load image; end
root = fileparts(fileparts(mfilename('fullpath')));
out = fullfile(root, 'results', 'visual'); if ~exist(out, 'dir'), mkdir(out); end
net = icinn_load(fullfile(root, 'models', 'icinn.mat'));
list = {'test_kodak/kodim23.png', 'test_bsd100/bsd_010.png', 'test_medical/med05_CR.png', 'test_medical/med03_MR.png'};
key = 20261127; s = 4;
for i = 1:numel(list)
    I = imread(fullfile(root, 'data', list{i}));
    nb = numel(I) / 4 / s^2; rand('seed', i); msg = rand(nb, 1) > 0.5;   %#ok<RAND>
    S = inncr_embed(I, msg, net, s, key);
    [m2, I2] = inncr_extract(S, net, s, key);
    [~, name] = fileparts(list{i});
    imwrite(I, fullfile(out, [name '_cover.png']));
    imwrite(S, fullfile(out, [name '_stego.png']));
    imwrite(uint8(min(255, 20 * abs(double(S) - double(I)))), fullfile(out, [name '_diff20.png']));
    imwrite(I2, fullfile(out, [name '_recovered.png']));
    fprintf('%s: PSNR %.2f dB, exact %d, BER %.4f\n', name, metric_psnr(S, I), isequal(I2, I), mean(m2(:) ~= msg(:)));
end
end
