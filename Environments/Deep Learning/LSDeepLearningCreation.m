function [noisytrainX,noisytrainY,noisytest,testsetY] = LSDeepLearningCreation(W,data,epsilon,gamma,testsetsize)

index = randsample(1:length(data),length(W)+testsetsize);
randdata = data(index,:);
trainset = randdata(1:length(W),:);
testsetX = randdata(length(W)+1:end,1:size(data,2)-1);
testsetY = randdata(length(W)+1:end,size(data,2));

B = W * trainset;
A = B(1:size(trainset)/3,:);
mu = max(A,[],'all');
v = min(A,[],'all');
a = gamma*((2*A - mu - v)/(mu - v));
X = laplace(size(A,1),size(a,2),0,1/(epsilon/(1+exp(-gamma))));
astar = zeros(size(X));
[rowsize,colsize] = size(X);

for ii=1:rowsize
    for jj=1:colsize
        if X(ii,jj) >= 0
            astar(ii,jj) = (1 - 1/(1 + exp(-a(ii,jj)))) * X(ii,jj);
        else
            astar(ii,jj) = (1/(1 + exp(-a(ii,jj)))) * X(ii,jj);
        end
    end
end

Ahat = A + astar;
Bhat = [Ahat; B((size(trainset)/3)+1:end,:)];
noisytrain = W' * Bhat;

dataresp = noisytrain(:,size(noisytrain,2));
dataresp(dataresp >= 0.5) = 1;
dataresp(dataresp < 0.5) = 0;
noisytrain = [noisytrain(:,1:size(noisytrain,2)-1) dataresp];
noisytrainX = noisytrain(:,1:size(noisytrain,2)-1);
noisytrainY = noisytrain(:,size(noisytrain,2));

%Add noise to testing set
w = round(log(testsetsize)/log(3));
Wt = wavsize(w);

Bt = Wt * testsetX;
At = Bt(1:size(testsetX)/3,:);
mut = max(At,[],'all');
vt = min(At,[],'all');
at = gamma*((2*At - mut - vt)/(mut - vt));
Xt = laplace(size(At,1),size(at,2),0,1/epsilon);
astart = zeros(size(Xt));
[rowsizet,colsizet] = size(Xt);

for iit=1:rowsizet
    for jjt=1:colsizet
        if Xt(iit,jjt) >= 0
            astart(iit,jjt) = (1 - 1/(1 + exp(-at(iit,jjt)))) * Xt(iit,jjt);
        else
            astart(iit,jjt) = (1/(1 + exp(-at(iit,jjt)))) * Xt(iit,jjt);
        end
    end
end

Ahatt = At + astart;
Bhatt = [Ahatt; Bt((size(testsetX)/3)+1:end,:)];
noisytest = Wt' * Bhatt;

csvwrite('2trainX82.csv',noisytrainX);
csvwrite('2trainY82.csv',noisytrainY);
csvwrite('2testX82.csv',noisytest);
csvwrite('2testY82.csv',testsetY);
end