function y  = laplace(m,n,mu,b)
% This generates a matrix of Laplace noise. 
u = rand(m,n) - 0.5;
y = mu - b*sign(u).*log(1 - 2*abs(u));
end