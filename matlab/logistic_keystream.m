function k = logistic_keystream(n, key)
% LOGISTIC_KEYSTREAM  n pseudo-random bits from the logistic map
% x(t+1) = r*x(t)*(1-x(t)), r = 3.99, x0 taken from the secret key.
% Used to scramble the message before embedding (without the key the
% extracted bits look random).
x = mod(key * 0.618033988749895, 1);
if x == 0, x = 0.123456789; end
r = 3.99;
for t = 1:200, x = r*x*(1-x); end       % discard the transient
k = false(n, 1);
for t = 1:n
    x = r*x*(1-x);
    k(t) = x > 0.5;
end
end
