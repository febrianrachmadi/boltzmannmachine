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

% This program trains Restricted Boltzmann Machine in which
% visible, binary, stochastic pixels are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.   
% The program assumes that the following variables are set externally:
% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 

%% Initialising learning variables
epsilonw      = 0.05;   % Learning rate for weights 
epsilonvb     = 0.05;   % Learning rate for biases of visible units 
epsilonhb     = 0.05;   % Learning rate for biases of hidden units 

CD = 1;   
weightcost  = 0.001;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;

poshidprobs = zeros(numcases,numhid);         % positive hidden probs
neghidprobs = zeros(numcases,numhid);         % negative hidden probs
posprods    = zeros(numvis,numhid);          % values after multiplied with positive probs
negprods    = zeros(numvis,numhid);          % values after multiplied with negative probs
vishidinc  = zeros(numvis,numhid);           % visble-hidden increment
hidbiasinc = zeros(1,numhid);                 % hidden biases increment
visbiasinc = zeros(1,numvis);                % visible biases increment
batchposhidprobs = zeros(numcases,numhid,numbatches);   % batch probs for positive hidden layer

%% Restarting the learning
if restart == 1
    restart = 0;

    % Initializing symmetric weights and biases. 
    vishid     = 0.001*randn(numvis, numhid);    % visible-hidden layers
    hidbiases  = zeros(1,numhid);                 % hidden layer's biases
    visbiases  = zeros(1,numvis);                % visible layer's biases
    epoch = 1;
    err_hist = zeros(1,maxepoch);
else
    fprintf('\nContinuing pretraining with additional %d epoch..\n', maxepoch);
    fprintf('Load fullmri_L%d_%d_%d.mat\n', layer_number, numvis, numhid);
    load(sprintf('fullmri_L%d_%d_%d.mat', layer_number, numvis, numhid));
    fprintf('Errsum: %6.1f\n', err_hist(end));
    
    % Initializing symmetric weights and biases. 
    vishid     = vishid;        % visible-hidden layers
    hidbiases  = hidbiases;     % hidden layer's biases
    visbiases  = visbiases;     % visible layer's biases
    maxepoch = maxepoch + epoch;    % continuing epoch
    err_hist_temp = zeros(1,maxepoch);
    err_hist_temp(1,1:epoch) = err_hist(1,:);
    err_hist = err_hist_temp;
    epoch = epoch + 1;
end

%% Learning loop
for epoch = epoch:maxepoch
    fprintf('epoch %d (%d batches)\r',epoch, numbatches);    
    errsum=0;
    for batch = 1:numbatches,
%         fprintf(1,'epoch %d batch %d\r',epoch,batch); 
        fprintf('.');
        if mod(batch, 50) == 0
            fprintf('\n');
        end

        visbias = repmat(visbiases,numcases,1);
        hidbias = repmat(2*hidbiases,numcases,1);   % hidden bias has weight 2 (DBM purpose)
        
        %% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data = batchdata(:,:,batch);
        data = data > rand(numcases,numvis);

        poshidprobs = 1./(1 + exp(-data*(2*vishid) - hidbias));     % vishid has weight 2 (DBM purpose)
        batchposhidprobs(:,:,batch) = poshidprobs;
        posprods    = data' * poshidprobs;
        poshidact   = sum(poshidprobs);
        posvisact = sum(data);

        %%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        poshidstates = poshidprobs > rand(numcases,numhid);
        negdata = 1./(1 + exp(-poshidstates*vishid' - visbias));
        negdata = negdata > rand(numcases,numvis); 
        neghidprobs = 1./(1 + exp(-negdata*(2*vishid) - hidbias));

        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata); 

        %%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %% Calculate error
        err = sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;

        %% Change momentum value
        if epoch > 5
            momentum = finalmomentum;
        else
            momentum = initialmomentum;
        end

        %% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        vishidinc = momentum * vishidinc + ...
                    epsilonw * ( (posprods-negprods)/numcases - weightcost*vishid);
        visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
        hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);

        % Learned information that would be saved
        vishid = vishid + vishidinc;            % Note: W
        visbiases = visbiases + visbiasinc;     % Note: L
        hidbiases = hidbiases + hidbiasinc;     % Note: J
        %%%%%%%%%%%%%%%% END OF UPDATES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
        %% Visualise generated data
        %    if rem(batch,600)==0  
        %      figure(1); 
        %      dispims(negdata',28,28);
        %      drawnow
        %    end  
    end
    fprintf('epoch %4i (errsum %6.1f)  \n', epoch, errsum); 
    err_hist(1,epoch) = errsum;
    
    if mod(epoch,5) == 0
        fprintf('Saving %d-st layer data to a file..\n', layer_number);
        save(sprintf('fullmri_L%d_%d_%d.mat', layer_number, numvis, numhid),...
            'vishid','visbiases','hidbiases','epoch','err_hist');
    end

end

fprintf('Saving %d-st layer data to a file..\n', layer_number);
save(sprintf('fullmri_L%d_%d_%d.mat', layer_number, numvis, numhid),...
    'vishid','visbiases','hidbiases','epoch','errsum','err_hist');
