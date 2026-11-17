function x = haar_idwt(Y)
% HAAR_IDWT  Inverse of haar_dwt
ll = Y(:,:,1); hl = Y(:,:,2); lh = Y(:,:,3); hh = Y(:,:,4);
x = zeros(2*size(Y,1), 2*size(Y,2));
x(1:2:end, 1:2:end) = (ll - hl - lh + hh)/2;
x(1:2:end, 2:2:end) = (ll + hl - lh - hh)/2;
x(2:2:end, 1:2:end) = (ll - hl + lh - hh)/2;
x(2:2:end, 2:2:end) = (ll + hl + lh + hh)/2;
end
