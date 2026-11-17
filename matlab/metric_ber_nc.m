function [ber, nc] = metric_ber_nc(b0, b1)
% METRIC_BER_NC  bit error rate and normalised correlation between the
% embedded bits b0 and the extracted bits b1 (NC uses the +-1 form)
b0 = double(b0(:)); b1 = double(b1(:));
ber = mean(b0 ~= b1);
x = 2*b0 - 1; y = 2*b1 - 1;
nc = sum(x .* y) / sqrt(sum(x.^2) * sum(y.^2));
end
