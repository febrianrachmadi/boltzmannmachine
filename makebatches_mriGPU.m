% Version 1.000
%
% Code provided by Ruslan Salakhutdinov 
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

%% Scaling each feature to [0...1]
minimums = min(features, [], 1);
maximums = max(features, [], 1);
ranges = maximums - minimums;
scaling_properties = [minimums; maximums; ranges;];
save(sprintf('%s/SCALING_Features_Kernel_111111.mat', data_path),'scaling_properties');
features = (features - repmat(minimums, size(features, 1), 1)) ./ repmat(ranges, size(features, 1), 1);
num_data = size(features,1);

%% Create training data batches
num_train = 200000;
Y_trn = [labels(1:num_train,:), ~labels(1:num_train,:)];

totnum = num_train;
fprintf(1, 'Size of the training dataset= %5d \n', totnum);
totnum_train = totnum;

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims   = size(features(1:num_train,:),2);
numlabels = size(Y_trn,2);
batchsize = 100;
batchdata = zeros(batchsize, numdims, numbatches,'single','gpuArray');
batchtargets = zeros(batchsize, numlabels, numbatches,'single','gpuArray');

for b=1:numbatches
  batchdata(:,:,b) = features(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = Y_trn(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear Y_trn;

%% Create test data batches
num_test = 6000;
X_test = features(num_train+1:num_train+num_test,:);
Y_test = [labels(num_train+1:num_train+num_test,:), ~labels(num_train+1:num_train+num_test,:)];

totnum = num_test;
fprintf(1, 'Size of the test dataset= %5d \n', totnum);
totnum_test = totnum;

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims   = size(X_test,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, ceil(numbatches),'single','gpuArray');
testbatchtargets = zeros(batchsize, numlabels, ceil(numbatches),'single','gpuArray');

for b=1:numbatches
  testbatchdata(:,:,b) = X_test(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = Y_test(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear X_test Y_test features labels;

%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



