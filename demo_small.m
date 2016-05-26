randn('state',100);
rand('state',100);
warning off

% addpath(genpath('/home/s1467961/V/frachmadi_ADNI/febdissdata/DBM/DBM_MRI/'));

clear all
close all

fprintf('Loading data.. \n');
dir_path = '/home/s1467961/V/frachmadi_ADNI/febdissdata/segmentation/groundtruth_ready';
load(sprintf('features_dl_all_120k_rand_flair.mat'))

fprintf('Pretraining a Deep Boltzmann Machine. \n');
makebatches_mri; 
[numcases, numdims, numbatches]=size(batchdata);
% [numcases, numdims, numbatches]=size(testbatchdata);

layers = [1331, 10, 15, 10, 15];
numlayers = numel(layers)-1;
numlab = 2;

for layer_number = 1 : numlayers
    if layer_number == 1    % first layer
        numvis = layers(layer_number);   % input layer
        numhid = layers(layer_number+1); % hidden layer
        maxepoch = 3;
        fprintf('Pretraining Layer %d with RBM: %d-%d \n', layer_number,numvis,numhid);
        restart = 1;
        rbm
    elseif layer_number == numlayers    % last layer
        numvis = layers(layer_number);
        numhid = layers(layer_number+1);        % input layer
        maxepoch = 3;
        fprintf('Pretraining Layer %d with RBM: %d-%d \n', layer_number,numvis,numhid);
        restart = 1;
        rbm_top_class
    else    % middle layers
        numvis = layers(layer_number-1);
        numhid = layers(layer_number);        % input layer
        numpen = layers(layer_number+1);        % hidden layer
        maxepoch = 3;
        fprintf('Pretraining Layer %d with RBM: %d-%d \n', layer_number,numhid,numpen);
        restart = 1;
        rbm_mid
    end
end

%%%%%% Training two-layer Boltzmann machine %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layers_left = [1331, 10, 15, 10, 15];
layers_right = [1331, 10, 15, 10, 15];
layers = [layers_left, layers_right];
numsharedunits = 30;
num_layers_left = numel(layers_left);
num_layers_right = numel(layers_right);

batchdata_left = batchdata;
batchdata_right = batchdata;

maxepoch = 3;
num_mf_updates = 10;
dbm_verbose = 0;
fprintf('Learning a Deep Bolztamnn Machine. \n');
restart = 1;
% dbm_mf_all_layers_class
mdbm_mf_all_layers_class


