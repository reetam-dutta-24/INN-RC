function [cand, pred, order, mir] = pee_setup(J, setId)
% PEE_SETUP  Candidate pixels of one checkerboard set, their rhombus
% prediction and the smooth-to-busy visiting order.
% Only pixels of the OTHER set are used, so the decoder gets the same values.
[H, W] = size(J);
[C, R] = meshgrid(2:W-1, 2:H-1);
sel  = mod(R + C, 2) == setId;
r = R(sel); c = C(sel);
cand = sub2ind([H W], r, c);
up = J(cand - 1); dn = J(cand + 1); lf = J(cand - H); rt = J(cand + H);
pred = floor((up + dn + lf + rt) / 4 + 0.5);
% Dark pixels (prediction < 128) use the usual bins 0/-1 and move up for
% bit 1; bright pixels use the mirrored rule (bins 0/+1, move down), so a
% flat white area behaves like a flat black one and never overflows.
% Predictions just inside the ends (1..4, 251..254) are left out, otherwise
% noisy black/white areas fill the overflow list. 0 and 255 are kept.
% The decoder computes the same predictions, so it makes the same choices.
keep = (pred == 0 | pred >= 5) & (pred <= 250 | pred == 255);
cand = cand(keep); pred = pred(keep);
up = up(keep); dn = dn(keep); lf = lf(keep); rt = rt(keep);
nb = [up dn lf rt];
cx = var(nb, 1, 2) + abs(up - dn) + abs(lf - rt);    % local complexity
[~, order] = sort(cx + (1:numel(cx))' * 1e-9);          % stable, deterministic
mir = pred >= 128;
end
