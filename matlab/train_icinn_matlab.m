function train_icinn_matlab(outFile, iters)
% TRAIN_ICINN_MATLAB  Trains the IC-INN with the Deep Learning Toolbox
% (custom training loop: dlarray / dlfeval / dlgradient / adamupdate).
% It follows training/train_icinn.py step by step and saves the weights in
% the same layout, so icinn_load() can read either file.
%
%   train_icinn_matlab('../models/icinn_matlab.mat', 12000)
%
% Needs: Deep Learning Toolbox. A GPU is used automatically if available.

if nargin < 1, outFile = fullfile('..', 'models', 'icinn_matlab.mat'); end
if nargin < 2, iters = 12000; end
root = fileparts(fileparts(mfilename('fullpath')));
rng(1);

% ---- hyper-parameters (same as the Python run used in the paper) ----
P.batch = 12; P.patch = 64; P.lr = 1e-3; P.hid = 32; P.nBlocks = 4;
P.wImg = 1; P.wMsg = 10; P.wOf = 5; P.warm = 0.25; P.sigma = 2;
scales = [2 4 8];

% ---- data: Urban100 grayscale, first 8 images kept for validation ----
f = dir(fullfile(root, 'data', 'train_urban100', '*.png'));
imgs = cell(numel(f), 1);
for i = 1:numel(f), imgs{i} = single(imread(fullfile(f(i).folder, f(i).name))); end
train = imgs(9:end);

% ---- parameters ----
for k = 1:P.nBlocks
    net.phi(k).W1 = initW(3, 3, 3, P.hid);     net.phi(k).b1 = zeros(1, 1, P.hid, 'single');
    net.phi(k).W2 = initW(3, 3, P.hid, P.hid); net.phi(k).b2 = zeros(1, 1, P.hid, 'single');
    net.phi(k).W3 = 0.05 * randn(3, 3, P.hid, 2, 'single'); net.phi(k).b3 = zeros(1, 1, 2, 'single');
end
net.dec.W1 = initW(3, 3, 4, P.hid);     net.dec.b1 = zeros(1, 1, P.hid, 'single');
net.dec.W2 = initW(3, 3, P.hid, P.hid); net.dec.b2 = zeros(1, 1, P.hid, 'single');
net.dec.W3 = initW(3, 3, P.hid, P.hid); net.dec.b3 = zeros(1, 1, P.hid, 'single');
net.dec.W4 = initW(3, 3, P.hid, 1);     net.dec.b4 = zeros(1, 1, 1, 'single');
net = dlupdate(@dlarray, net);
if canUseGPU, net = dlupdate(@gpuArray, net); end

avgG = []; avgSqG = [];
lossHist = zeros(iters, 3);
for it = 1:iters
    s = scales(randi(numel(scales)));
    x = randomPatches(train, P.batch, P.patch);
    bits = single(rand(P.patch/2/s, P.patch/2/s, 1, P.batch) > 0.5);
    Mmap = repelem(2*bits - 1, s, s, 1, 1);                % message map, H/2 x W/2
    x = dlarray(x, 'SSCB'); b = dlarray(bits, 'SSCB'); Mmap = dlarray(Mmap, 'SSCB');
    if canUseGPU, x = gpuArray(x); b = gpuArray(b); Mmap = gpuArray(Mmap); end
    ramp = min(1, max(0, (it - 0.1*iters) / ((P.warm + 0.25) * iters)));
    w = [P.wImg * max(ramp^2, 0.002), P.wMsg, P.wOf * max(ramp, 0.05)];
    useNoise = it < iters/2;
    [loss, grad, parts] = dlfeval(@modelLoss, net, x, b, Mmap, s, w, useNoise, P.sigma);
    lr = P.lr * (0.02 + 0.98 * 0.5 * (1 + cos(pi * it / iters)));   % cosine schedule
    [net, avgG, avgSqG] = adamupdate(net, grad, avgG, avgSqG, it, lr);
    lossHist(it, :) = double(gather(extractdata(parts)));
    if mod(it, 250) == 0
        fprintf('it %5d  s=%d  loss %.4f  mse %.3f (PSNR %.2f)  bce %.4f\n', it, s, ...
            gather(extractdata(loss)), lossHist(it,1), 10*log10(255^2 / lossHist(it,1)), lossHist(it,2));
    end
end

% ---- export in the PyTorch layout (out, in, 3, 3) ----
S = struct('n_blocks', P.nBlocks, 'hid', P.hid);
for k = 1:P.nBlocks
    for l = 1:3
        S.(sprintf('phis_%d_c%d_weight', k-1, l)) = toTorch(net.phi(k).(sprintf('W%d', l)));
        S.(sprintf('phis_%d_c%d_bias', k-1, l))   = squeeze(double(gather(extractdata(net.phi(k).(sprintf('b%d', l))))));
    end
end
for l = 1:4
    S.(sprintf('dec_c%d_weight', l)) = toTorch(net.dec.(sprintf('W%d', l)));
    S.(sprintf('dec_c%d_bias', l))   = squeeze(double(gather(extractdata(net.dec.(sprintf('b%d', l))))));
end
save(outFile, '-struct', 'S');
save(strrep(outFile, '.mat', '_loss.mat'), 'lossHist');
fprintf('saved %s\n', outFile);
end

% ======================================================================
function [loss, grad, parts] = modelLoss(net, x, b, M, s, w, useNoise, sigma)
yt = embedTrain(net, x, M, useNoise);
y = yt + stopGrad(min(max(yt, 0), 255) - yt);          % clip with straight-through gradient
yn = noiseLayer(y, sigma);
yn = yn + stopGrad(min(max(round(yn), 0), 255) - yn);
L = decode(net, yn);
L = avgpool(L, s, 'Stride', s);
bce = mean(max(L, 0) - L .* b + log(1 + exp(-abs(L))), 'all');
mse = mean((y - x).^2, 'all');
of = mean(max(-yt, 0) + max(yt - 255, 0), 'all');
loss = w(1)*mse + w(2)*bce + w(3)*of;
grad = dlgradient(loss, net);
parts = [mse, bce, of];
end

function yt = embedTrain(net, x, M, useNoise)
a = x(1:2:end, 1:2:end, :, :); bb = x(1:2:end, 2:2:end, :, :);
c = x(2:2:end, 1:2:end, :, :); d = x(2:2:end, 2:2:end, :, :);
xa = cat(3, a, d); xb = cat(3, bb, c);
for k = 1:numel(net.phi)
    if mod(k, 2) == 1
        xb = xb + roundTrain(phi(net.phi(k), xa, M), useNoise);
    else
        xa = xa + roundTrain(phi(net.phi(k), xb, M), useNoise);
    end
end
yt = x;   % same size/format, filled below
yt(1:2:end, 1:2:end, :, :) = xa(:, :, 1, :); yt(2:2:end, 2:2:end, :, :) = xa(:, :, 2, :);
yt(1:2:end, 2:2:end, :, :) = xb(:, :, 1, :); yt(2:2:end, 1:2:end, :, :) = xb(:, :, 2, :);
end

function v = roundTrain(v, useNoise)
if useNoise
    v = v + (rand(size(v), 'like', extractdata(v)) - 0.5);    % uniform noise proxy
else
    v = v + stopGrad(round(v) - v);                            % straight-through rounding
end
end

function off = phi(p, half, M)
h = cat(3, half / 255, M);
h = leakyrelu(dlconv(h, p.W1, p.b1, 'Padding', 'same'), 0.2);
h = leakyrelu(dlconv(h, p.W2, p.b2, 'Padding', 'same'), 0.2);
off = 16 * tanh(0.5 * dlconv(h, p.W3, p.b3, 'Padding', 'same'));
end

function L = decode(net, y)
y = y / 255;
h = cat(3, y(1:2:end, 1:2:end, :, :), y(1:2:end, 2:2:end, :, :), y(2:2:end, 1:2:end, :, :), y(2:2:end, 2:2:end, :, :));
h = leakyrelu(dlconv(h, net.dec.W1, net.dec.b1, 'Padding', 'same'), 0.2);
h = leakyrelu(dlconv(h, net.dec.W2, net.dec.b2, 'Padding', 'same'), 0.2);
h = leakyrelu(dlconv(h, net.dec.W3, net.dec.b3, 'Padding', 'same'), 0.2);
L = dlconv(h, net.dec.W4, net.dec.b4, 'Padding', 'same');
end

function y = noiseLayer(y, sigma)
r = rand;
if r < 0.45, return; end
if r < 0.75
    y = y + randn(size(y), 'like', extractdata(y)) * (rand * sigma);
elseif r < 0.9
    m = single(rand(size(y)) < rand * 0.02); sp = single(rand(size(y)) < 0.5) * 255;
    y = y .* (1 - m) + sp .* m;
else
    k = [1 2 1]' * [1 2 1] / 16;
    yp = cat(1, y(1, :, :, :), y, y(end, :, :, :)); yp = cat(2, yp(:, 1, :, :), yp, yp(:, end, :, :));
    y = dlconv(yp, single(k), 0);
end
end

function d = stopGrad(d)
d = dlarray(extractdata(d), dims(d));     % constant: no gradient flows through it
end

function W = initW(kh, kw, cin, cout)
W = randn(kh, kw, cin, cout, 'single') * sqrt(2 / (kh * kw * cin));   % He initialisation
end

function x = randomPatches(imgs, n, s)
x = zeros(s, s, 1, n, 'single');
for i = 1:n
    im = imgs{randi(numel(imgs))};
    r = randi(size(im, 1) - s + 1); c = randi(size(im, 2) - s + 1);
    q = im(r:r+s-1, c:c+s-1);
    if rand < 0.5, q = fliplr(q); end
    if rand < 0.5, q = flipud(q); end
    x(:, :, 1, i) = q;
end
end

function T = toTorch(W)
% MATLAB (kh, kw, in, out) -> PyTorch (out, in, kh, kw)
T = permute(double(gather(extractdata(W))), [4 3 1 2]);
end
