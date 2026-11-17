function Y = haar_dwt(x)
% HAAR_DWT  One level orthonormal Haar transform -> H/2 x W/2 x 4 (LL HL LH HH)
a = x(1:2:end, 1:2:end); b = x(1:2:end, 2:2:end);
c = x(2:2:end, 1:2:end); d = x(2:2:end, 2:2:end);
Y = cat(3, (a+b+c+d)/2, (-a+b-c+d)/2, (-a-b+c+d)/2, (a-b-c+d)/2);
end
