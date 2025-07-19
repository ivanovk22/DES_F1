function x = truncexprnd(lambda, a, b, L)
    U = rand(1, L);
    Z = 1 - exp(-lambda * (b - a));
    x = -log(1 - U * Z) / lambda + a;
end





