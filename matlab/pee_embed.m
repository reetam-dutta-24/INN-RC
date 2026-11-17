function [J, info] = pee_embed(I, bits, maxLayers)
% PEE_EMBED  Multi-layer rhombus prediction-error expansion (reversible).
%
%   [J, info] = pee_embed(I, bits)   hides the bit vector BITS in the 8-bit
%   image I and returns the marked image J. pee_extract(J) gives back both
%   BITS and I exactly.
%
%   Each layer = 2 passes (cross pixels, then dot pixels, checkerboard).
%   In a pass every pixel is predicted from its 4 neighbours of the other
%   set (which are not touched in that pass), errors 0 and -1 are expanded,
%   the rest are shifted by one. Pixels are visited from smooth to busy
%   (sorted by local variance), like Sachnev et al. 2009.
%   Pixels that could overflow (0 or 255) are pulled in by one and their
%   positions are written into the header of that pass.
%
%   Pass payload = [nBits(24) | moreLayers(1) | nList(18) | list | chunk]
%   (the list is stored as Elias-gamma coded gaps between pixel indices)

if nargin < 3, maxLayers = 10; end
J = double(I);
bits = logical(bits(:)');
[H, W] = size(J);
hdrLen = 24 + 1 + 18;
pos = 0; nb = numel(bits);
layer = 0; info.passBits = [];
while true
    layer = layer + 1;
    if layer > maxLayers
        error('pee_embed:capacity', 'payload too large: %d of %d bits left after %d layers', nb - pos, nb, maxLayers);
    end
    for setId = [0 1]
        [cand, pred, order] = pee_setup(J, setId);
        c = cand(order); pr = pred(order);
        v0 = J(c);
        % only pixels that are actually visited can overflow, so the list is
        % built for the visited prefix; iterate because the list length
        % itself changes how far we have to go
        list = zeros(0,1);
        for it = 1:20
            v = v0;
            e = v - pr;
            riskAll = (v == 0 & e <= -1) | (v == 255 & e >= 0);
            listBits = gamma_encode(diff([0; sort(list(:))])');
            fixed = hdrLen + numel(listBits);
            v(ismember(c, list) & v == 0) = 1; v(ismember(c, list) & v == 255) = 254;
            e = v - pr;
            expandable = (e == 0 | e == -1);
            capTotal = sum(expandable);
            room = max(capTotal - fixed, 0);
            take = min(room, nb - pos);
            need = fixed + take;
            cum = cumsum(expandable);
            stopAt = find(cum >= need, 1);
            if isempty(stopAt), stopAt = numel(e); end
            newList = sort(c(riskAll(1:stopAt)));
            if all(ismember(newList, list)), break; end   % list covers every visited risky pixel
            list = union(list(:), newList(:));
        end
        if ~all(ismember(newList, list))
            error('pee_embed:overflow', 'overflow list did not converge');
        end
        if need > capTotal
            error('pee_embed:capacity', 'capacity exceeded at layer %d: %d embeddable bits, %d needed for header/overflow list (%d pixels)', layer, capTotal, fixed, numel(list));
        end
        J(c) = v;                                   % apply the pre-adjustment
        more = (layer > 1) && (setId == 1);
        payload = [dec2bin_vec(need, 24), more, dec2bin_vec(numel(list), 18), ...
                   listBits, bits(pos+1:pos+take)];
        pos = pos + take;
        % embed sequentially in sorted order until payload is used up
        k = 1:stopAt;
        ek = e(k); exk = expandable(k);
        b = zeros(size(ek));
        b(exk) = payload(1:sum(exk));
        en = ek;
        en(ek >= 1)  = ek(ek >= 1) + 1;
        en(ek <= -2) = ek(ek <= -2) - 1;
        en(ek == 0)  = b(ek == 0);
        en(ek == -1) = -1 - b(ek == -1);
        J(c(k)) = v(k) - ek + en;
        info.passBits(end+1) = numel(payload);
    end
    if pos >= nb, break; end
end
info.layers = layer;
info.bpp = nb / (H*W);
J = uint8(J);
end

function b = dec2bin_vec(n, w)
% row(s) of W bits, MSB first, for each entry of n
n = double(n(:));
b = logical(mod(floor(n ./ 2.^(w-1:-1:0)), 2));
if isempty(n), b = false(0, w); end
end
