function Y = conv3x3(X, W, b)
% CONV3X3  3x3 convolution with zero padding 1, same as PyTorch Conv2d
% (cross-correlation). X: H x W x Cin, W: Cout x Cin x 3 x 3, b: Cout.
% Implemented as one matrix product (im2col) so it is fast in MATLAB.
[h, w, cin] = size(X);
cout = size(W, 1);
if cin == 1 && ndims(W) == 3, W = reshape(W, cout, 1, 3, 3); end
Xp = zeros(h + 2, w + 2, cin);
Xp(2:end-1, 2:end-1, :) = X;
P = zeros(h*w, 9*cin);
Wm = zeros(9*cin, cout);
k = 0;
for u = 1:3
    for v = 1:3
        P(:, k*cin + (1:cin)) = reshape(Xp(u:u+h-1, v:v+w-1, :), h*w, cin);
        Wm(k*cin + (1:cin), :) = reshape(W(:, :, u, v), cout, cin)';
        k = k + 1;
    end
end
Y = reshape(P * Wm + reshape(b, 1, []), h, w, cout);
end
