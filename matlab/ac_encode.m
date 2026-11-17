function out = ac_encode(bits, ctx, nctx)
% AC_ENCODE  Adaptive binary arithmetic coder (32-bit integer version of the
% Witten-Neal-Cleary coder). Every bit is coded with the adaptive model of
% its context ctx(i) in 1..nctx. Returns the code as a logical row vector.
%
% All arithmetic stays below 2^53 so it is exact in double precision.

bits = logical(bits(:)); ctx = ctx(:);
TOP = 2^32 - 1; HALF = 2^31; QTR = 2^30;
c0 = ones(nctx, 1); c1 = ones(nctx, 1);          % symbol counts per context
low = 0; high = TOP; pending = 0;
out = false(1, numel(bits) + 64); n = 0;
for i = 1:numel(bits)
    k = ctx(i);
    rng = high - low + 1;
    split = low + floor(rng * c0(k) / (c0(k) + c1(k))) - 1;
    if bits(i)
        low = split + 1;  c1(k) = c1(k) + 1;
    else
        high = split;     c0(k) = c0(k) + 1;
    end
    if c0(k) + c1(k) > 4096                       % keep the model adaptive
        c0(k) = ceil(c0(k) / 2); c1(k) = ceil(c1(k) / 2);
    end
    while true
        if high < HALF
            [out, n] = put(out, n, false, pending); pending = 0;
        elseif low >= HALF
            [out, n] = put(out, n, true, pending); pending = 0;
            low = low - HALF; high = high - HALF;
        elseif low >= QTR && high < 3*QTR
            pending = pending + 1;
            low = low - QTR; high = high - QTR;
        else
            break;
        end
        low = 2*low; high = 2*high + 1;
    end
end
pending = pending + 1;
if low < QTR
    [out, n] = put(out, n, false, pending);
else
    [out, n] = put(out, n, true, pending);
end
out = out(1:n);
end

function [out, n] = put(out, n, b, pending)
need = n + 1 + pending;
if need > numel(out), out(end + max(1024, need - numel(out))) = false; end
out(n+1) = b;
out(n+2:n+1+pending) = ~b;
n = need;
end
