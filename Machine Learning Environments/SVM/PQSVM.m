function [averagepercentage] = PQSVM(W,data,epsilon,trials,testsetsize)
matrix = zeros(trials,1);

for i = 1:trials

index = randsample(1:length(data),length(W)+testsetsize);
randdata = data(index,:);
train = randdata(1:length(W),:);
testsetX = randdata(length(W)+1:end,1:size(data,2)-1);
testsetY = randdata(length(W)+1:end,size(data,2));
    
B = W * train;
A = B(1:size(train,1)/3,:);

mu = max(A(:));
v = min(A(:));
theta = zeros(size(A));

for k = 1:size(train,2)
    for L = 1:size(train,1)/3
        theta(L,k) = (pi*((A(L,k) + mu - 2*v)))/(6*(mu - v));
    end
end

X = laplace(size(train,1)/3,size(train,2),0,2/epsilon);

mu2 = max(X,[],'all');
v2 = min(X,[],'all');
alpha = pi*((X + mu2 - 2*v2)/(6*(mu2 - v2)));

K = rand(size(train,1)/3,size(train,2));

thetae = zeros(size(X));

for ii=1:size(train,1)/3
    for jj=1:size(train,2)
        if K(ii,jj) >= 0.5
            thetae(ii,jj) = acos(cos(theta(ii,jj)+0.1*cos(alpha(ii,jj))));
        else
            thetae(ii,jj) = asin(sin(theta(ii,jj))+0.1*sin(alpha(ii,jj)));
        end
    end
end

Astar = zeros(size(A));
for m = 1:size(train,1)/3
    for n = 1:size(train,2)
        Astar(m,n) = (6/pi)*(mu-v)*thetae(m,n) - mu + 2*v;
    end
end

Bstar = [Astar; B(size(train,1)/3 + 1:end,:)];
noisytrain = W' * Bstar;

noistrainY = noisytrain(:,size(noisytrain,2));
noistrainY(noistrainY >= 0.5) = 1;
noistrainY(noistrainY < 0.5) = 0;
noisytrain = [noisytrain(:,1:size(noisytrain,2)-1) noistrainY];
noisytrainX = noisytrain(:,1:size(noisytrain,2)-1);
noisytrainY = noisytrain(:,size(noisytrain,2));

%Add noise to testing set
w = round(log(testsetsize)/log(3));
Wt = wavsize(w);
Bt = Wt * testsetX;
At = Bt(1:size(testsetX,1)/3,:);

mut = max(At(:));
vt = min(At(:));
thetat = zeros(size(At));

for kt = 1:size(testsetX,2)
    for Lt = 1:size(testsetX,1)/3
        thetat(Lt,kt) = (pi*((At(Lt,kt) + mut - 2*vt)))/(6*(mut - vt));
    end
end

Xt = laplace(size(testsetX,1)/3,size(testsetX,2),0,2/epsilon);

mu2t = max(Xt,[],'all');
v2t = min(Xt,[],'all');
alphat = pi*((Xt + mu2t - 2*v2t)/(6*(mu2t - v2t)));

Kt = rand(size(testsetX,1)/3,size(testsetX,2));

thetaet = zeros(size(Xt));

for iit=1:size(testsetX,1)/3
    for jjt=1:size(testsetX,2)
        if Kt(iit,jjt) >= 0.5
            thetaet(iit,jjt) = acos(cos(thetat(iit,jjt)+0.1*cos(alphat(iit,jjt))));
        else
            thetaet(iit,jjt) = asin(sin(thetat(iit,jjt))+0.1*sin(alphat(iit,jjt)));
        end
    end
end

Astart = zeros(size(At));
for mt = 1:size(testsetX,1)/3
    for nt = 1:size(testsetX,2)
        Astart(mt,nt) = (6/pi)*(mut-vt)*thetaet(mt,nt) - mut + 2*vt;
    end
end

Bstart = [Astart; Bt(size(testsetX,1)/3 + 1:end,:)];
noisytest = Wt' * Bstart;

%Define & train the classifier
mdl = fitckernel(noisytrainX, noisytrainY);

%Testing set
resphat = predict(mdl,noisytest);
resphatrounded = round(resphat);


diff = testsetY - resphatrounded;
right = sum(diff(:) == 0);
percentage = right/testsetsize;

matrix(i,1) = percentage;
end
averagepercentage = 100*mean(matrix);
end