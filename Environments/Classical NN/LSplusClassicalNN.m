function [averagepercentage] = LSplusClassicalNN(W,data,epsilon,gamma,trials,trainsetsize,testsetsize)
matrix = zeros(trials,1);

for I = 1:trials
    
index = randsample(1:length(data),trainsetsize+testsetsize);
randdata = data(index,:);
trainset = randdata(1:trainsetsize,:);
testsetX = randdata(trainsetsize+1:end,1:size(data,2)-1);
testsetY = randdata(trainsetsize+1:end,size(data,2))';

noisytrain = zeros(size(trainset));
m = length(trainset);
w = length(W);

%Waveletblock
for i = 1:m/w
block = trainset(1+w*(i-1):w*i,:);

%Algorithm
B = W * block;
A = B(1:size(block)/3,:);
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
Bhat = [Ahat; B((size(block)/3)+1:end,:)];
noisyblock = W' * Bhat;

noisyblockresp = noisyblock(:,size(noisyblock,2));
noisyblockresp(noisyblockresp >= 0.5) = 1;
noisyblockresp(noisyblockresp < 0.5) = 0;
noisyblock = [noisyblock(:,1:size(noisyblock,2)-1) noisyblockresp];

%Waveletblockreverse
noisytrain(1+w*(i-1):w*i,:) = noisyblock;
end

%ADD NOISE TO TESTING

noisytest = zeros(size(testsetX));
mt = length(testsetX);
wt = length(W);

%Waveletblock
for it = 1:mt/wt
blockt = testsetX(1+wt*(it-1):wt*it,:);

%Algorithm
Bt = W * blockt;
At = Bt(1:size(blockt)/3,:);
mut = max(At,[],'all');
vt = min(At,[],'all');
at = gamma*((2*At - mut - vt)/(mut - vt));
Xt = laplace(size(At,1),size(at,2),0,1/(epsilon/(1+exp(-gamma))));
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
Bhatt = [Ahatt; Bt((size(blockt)/3)+1:end,:)];
noisytest(1+wt*(it-1):wt*it,:) = W' * Bhatt;
end


noisytrainX = noisytrain(:,1:size(noisytrain,2)-1)';
noisytrainY = noisytrain(:,size(noisytrain,2))';

noisytest = noisytest';

%Define & train the classifier
net = feedforwardnet(10);
net = configure(net,noisytrainX,noisytrainY);

trainednet = train(net,noisytrainX,noisytrainY);

%Testing set
resphat = trainednet(noisytest);
resphatrounded = round(resphat);

diff = testsetY - resphatrounded;
right = sum(diff(:) == 0);
percentage = right/testsetsize;

matrix(I,1) = percentage;
end

averagepercentage = 100*mean(matrix);
end