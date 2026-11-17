function [cand, pred, order] = pee_setup(J, setId)
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
% flat saturated areas (sky, white film) would push 255 -> 256, so pixels
% predicted as 254/255 are left out. Same for prediction 1 on black
% backgrounds (0 with error -1 would go to -1). Prediction 0 is safe.
% The decoder computes the same predictions, so it skips the same pixels.
keep = pred <= 253 & pred ~= 1;
cand = cand(keep); pred = pred(keep);
up = up(keep); dn = dn(keep); lf = lf(keep); rt = rt(keep);
nb = [up dn lf rt];
cx = var(nb, 1, 2) + abs(up - dn) + abs(lf - rt);    % local complexity
[~, order] = sort(cx + (1:numel(cx))' * 1e-9);          % stable, deterministic
end
