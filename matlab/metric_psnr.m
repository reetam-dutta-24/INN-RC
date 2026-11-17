function p = metric_psnr(a, b)
% METRIC_PSNR  PSNR in dB between two 8-bit images (Inf if identical)
mse = mean((double(a(:)) - double(b(:))).^2);
p = 10*log10(255^2 / mse);
end
