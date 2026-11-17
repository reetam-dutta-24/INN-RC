function J = apply_attack(I, type, param)
% APPLY_ATTACK  Distortions used for the robustness test
%   'gauss' : additive Gaussian noise, std = param grey levels
%   'sp'    : salt & pepper, density = param
%   'jpeg'  : JPEG compression, quality = param
%   'none'
I = double(I);
switch type
    case 'gauss'
        J = I + param * randn(size(I));
    case 'sp'
        J = I; r = rand(size(I));
        J(r < param/2) = 0; J(r > 1 - param/2) = 255;
    case 'jpeg'
        f = [tempname '.jpg'];
        imwrite(uint8(I), f, 'Quality', param);
        J = double(imread(f)); delete(f);
    otherwise
        J = I;
end
J = uint8(min(max(round(J), 0), 255));
end
