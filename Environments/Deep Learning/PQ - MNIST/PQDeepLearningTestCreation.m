function [datatest] = PQDeepLearningTestCreation(test,epsilon)
datatest = zeros(10000,27,27);
% noisytestX = zeros(10000,729);
W = wavsize(3);

for i = 1:10000
kk = reshape(test(i,:), 28, 28)';
J = kk(1:27,1:27);

B = W * J;
A = B(1:size(J,1)/3,:);

mu = max(A(:));
v = min(A(:));
theta = zeros(size(A));

for k = 1:size(J,2)
    for L = 1:size(J,1)/3
        theta(L,k) = (pi*((A(L,k) + mu - 2*v)))/(6*(mu - v));
    end
end

X = laplace(size(J,1)/3,size(J,2),0,2/epsilon);

mu2 = max(X,[],'all');
v2 = min(X,[],'all');
alpha = pi*((X + mu2 - 2*v2)/(6*(mu2 - v2)));

K = rand(size(J,1)/3,size(J,2));

thetae = zeros(size(X));

for ii=1:size(J,1)/3
    for jj=1:size(J,2)
        if K(ii,jj) >= 0.5
            thetae(ii,jj) = acos(cos(theta(ii,jj)+0.1*cos(alpha(ii,jj))));
        else
            thetae(ii,jj) = asin(sin(theta(ii,jj))+0.1*sin(alpha(ii,jj)));
        end
    end
end

Astar = zeros(size(A));
for m = 1:size(J,1)/3
    for n = 1:size(J,2)
        Astar(m,n) = (6/pi)*(mu-v)*thetae(m,n) - mu + 2*v;
    end
end

Bstar = [Astar; B(size(J,1)/3 + 1:end,:)];
datanew = W' * Bstar;

datatest(i,:,:) = datanew;

% datavector = reshape(datanew',1,729);
% noisytestX(i,:) = datavector;
end
end