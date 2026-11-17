function bits = ac_decode(code, ctx, nctx)
% AC_DECODE  Decoder for ac_encode. ctx must be the same context sequence
% that was used by the encoder (its length = number of bits to decode).

code = logical(code(:)'); ctx = ctx(:);
TOP = 2^32 - 1; HALF = 2^31; QTR = 2^30;
c0 = ones(nctx, 1); c1 = ones(nctx, 1);
low = 0; high = TOP;
code = [code false(1, 64)];                      % pad with zeros at the end
value = sum(double(code(1:32)) .* 2.^(31:-1:0));
p = 33;
bits = false(numel(ctx), 1);
for i = 1:numel(ctx)
    k = ctx(i);
    rng = high - low + 1;
    split = low + floor(rng * c0(k) / (c0(k) + c1(k))) - 1;
    if value > split
        bits(i) = true;  low = split + 1;  c1(k) = c1(k) + 1;
    else
        high = split;    c0(k) = c0(k) + 1;
    end
    if c0(k) + c1(k) > 4096
        c0(k) = ceil(c0(k) / 2); c1(k) = ceil(c1(k) / 2);
    end
    while true
        if high < HALF
            % nothing to subtract
        elseif low >= HALF
            low = low - HALF; high = high - HALF; value = value - HALF;
        elseif low >= QTR && high < 3*QTR
            low = low - QTR; high = high - QTR; value = value - QTR;
        else
            break;
        end
        low = 2*low; high = 2*high + 1;
        value = 2*value + code(p); p = p + 1;
        if p > numel(code), code(end+64) = false; end
    end
end
end
