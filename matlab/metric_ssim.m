function s = metric_ssim(a, b)
% METRIC_SSIM  Mean SSIM (Wang et al. 2004): 11x11 Gaussian window,
% sigma 1.5, K1 = 0.01, K2 = 0.03, L = 255. Same settings as MATLAB ssim().
a = double(a); b = double(b);
C1 = (0.01*255)^2; C2 = (0.03*255)^2;
[u, v] = meshgrid(-5:5); g = exp(-(u.^2 + v.^2) / (2*1.5^2)); g = g / sum(g(:));
f = @(x) conv2(x, g, 'valid');
ma = f(a); mb = f(b);
va = f(a.^2) - ma.^2; vb = f(b.^2) - mb.^2; cab = f(a.*b) - ma.*mb;
m = ((2*ma.*mb + C1) .* (2*cab + C2)) ./ ((ma.^2 + mb.^2 + C1) .* (va + vb + C2));
s = mean(m(:));
end
