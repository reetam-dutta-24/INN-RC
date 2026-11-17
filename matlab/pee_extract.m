function [bits, I] = pee_extract(J)
% PEE_EXTRACT  Inverse of pee_embed: returns the hidden bits and the
% original image (bit-exact).
I = double(J);
[H, W] = size(I);
chunks = {};
more = true;
while more
    for setId = [1 0]                      % undo dot pass first, then cross
        [cand, pred, order, mir] = pee_setup(I, setId);
        c = cand(order); mr = mir(order);
        v = I(c); e = v - pred(order);
        expandable = (~mr & e >= -2 & e <= 1) | (mr & e >= -1 & e <= 2);
        b = (~mr & (e == 1 | e == -2)) | (mr & (e == -1 | e == 2));
        % header: first 24 bits carried by the first expandable pixels
        exIdx = find(expandable);
        hdr = b(exIdx(1:24));
        n = bin2dec_vec(hdr);
        stopAt = exIdx(n);
        k = 1:stopAt;
        ek = e(k); mk = mr(k); sd = ~mk;
        eo = ek;
        eo(sd & ek >= 2)  = ek(sd & ek >= 2) - 1;
        eo(sd & ek <= -3) = ek(sd & ek <= -3) + 1;
        eo(sd & (ek == 0 | ek == 1))   = 0;
        eo(sd & (ek == -1 | ek == -2)) = -1;
        eo(mk & ek <= -2) = ek(mk & ek <= -2) + 1;
        eo(mk & ek >= 3)  = ek(mk & ek >= 3) - 1;
        eo(mk & (ek == -1 | ek == 0)) = 0;
        eo(mk & (ek == 1 | ek == 2))  = 1;
        I(c(k)) = v(k) - ek + eo;
        payload = b(exIdx(1:n))';
        if setId == 1, more = payload(25); end
        nList = bin2dec_vec(payload(26:43));
        p = 44;
        if nList > 0
            [gaps, p] = gamma_decode(payload, p, nList);
            list = cumsum(gaps);
            I(list(I(list) == 1)) = 0;  I(list(I(list) == 254)) = 255;
        end
        chunks{end+1} = payload(p:end); %#ok<AGROW>
    end
end
% chunks were collected last-pass-first
bits = logical([chunks{end:-1:1}]);
I = uint8(I);
end

function n = bin2dec_vec(B)
B = double(B);
if isvector(B), B = B(:)'; end
n = B * (2.^(size(B,2)-1:-1:0))';
end
