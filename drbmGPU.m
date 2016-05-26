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
epsilonw_init  = 0.05;   % Learning rate for weights 
epsilonvb_init = 0.05;   % Learning rate for biases of visible units 
epsilonhb_init = 0.05;   % Learning rate for biases of hidden units 

%parameters to set
CD          = 1;   
weightcost  = 0.001;        % parameter of weigth decay 
initialmomentum  = 0.5;     % initial value of the momentum
finalmomentum    = 0.9;     % value of the momentum after 5 epochs
alpha       = 0;            % weight of the generative learning
init_gamma  = 0.9;      % initial learning rate
tau         = 1.9;      % descending speed of gamma 
discrim     = 1;        % 1 to activate the discriminative learning, 0 otherwise
freeenerg   = 1;        % if discrim is zero but one want to compute the free energy

data        = zeros(numcases,numvis,'double','gpuArray');
label       = zeros(numcases,numlab,'double','gpuArray');

poshidprobs = zeros(numcases,numhid,'double','gpuArray');     % positive hidden probs
neghidprobs = zeros(numcases,numhid,'double','gpuArray');     % negative hidden probs

posprods    = zeros(numvis,numhid,'double','gpuArray');       % values after multiplied with positive probs
negprods    = zeros(numvis,numhid,'double','gpuArray');       % values after multiplied with negative probs

vishidinc  = zeros(numvis,numhid,'double','gpuArray');        % visble-hidden increment
labhidinc  = zeros(numlab,numhid,'double','gpuArray');        % labels-hidden increment

hidbiasinc = zeros(1,numhid,'double','gpuArray');             % hidden biases increment
visbiasinc = zeros(1,numvis,'double','gpuArray');             % visible biases increment
labbiasinc = zeros(1,numlab,'double','gpuArray');             % labels biases increment

%% Restarting the learning
if restart == 1
    restart = 0;
    epoch = 1;

    % Initializing symmetric weights and biases. 
    vishid     = 0.001*randn(numvis, numhid,'double','gpuArray');     % visible-hidden layers
    labhid     = 0.001*randn(numlab, numhid,'double','gpuArray');     % label-hidden layers
    
    hidbiases  = zeros(1,numhid,'double','gpuArray');                 % hidden layer's biases
    visbiases  = zeros(1,numvis,'double','gpuArray');                 % visible layer's biases
    labbiases  = zeros(1,numlab,'double','gpuArray');                 % label layer's biases
    
    err_hist = zeros(1,maxepoch,'double','gpuArray');
else
    fprintf('\nContinuing pretraining with additional %d epoch..\n', maxepoch);
    fprintf('Load DRBM_L%d_%d_%d.mat\n', layer_number, numvis, numhid);
    load(sprintf('%s/DRBM_L%d_%d_%d.mat', data_path, layer_number, numvis, numhid));
    fprintf('Errsum: %6.1f\n', err_hist(end));
    
    % Initializing symmetric weights and biases. 
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
    gamma = (init_gamma) / (1 + (epoch - 1) / tau);
    
    CD = ceil(epoch/20);
    epsilonw  = epsilonw_init  / CD;   % Learning rate for weights 
    epsilonvb = epsilonvb_init / CD;   % Learning rate for biases of visible units 
    epsilonhb = epsilonhb_init / CD;   % Learning rate for biases of hidden units 
    
    for batch = 1:numbatches,
        %% START POSITIVE PHASE - Generative %%
        visbias = repmat(visbiases,numcases,1);
        hidbias = repmat(2*hidbiases,numcases,1);   % hidden bias has weight 2 (DBM purpose)
        labbias = repmat(labbiases,numcases,1);
                
        data(:,:) = batchdata(:,:,batch);
        data = double(gt(data,rand(numcases,numvis)));
        label(:,:) = batchtargets(:,:,batch);
        
        poshidprobs = h_given_xy(data, label, vishid, labhid, hidbias, 1);
        poshidstates = double(gt(poshidprobs,rand(numcases,numhid)));

        %%% END OF POSITIVE PHASE  %%%

        %% START NEGATIVE PHASE  - Generative %%
        negvisprobs = x_given_h(poshidstates, vishid, visbias);
        neglabprobs = y_given_h(poshidstates, labhid, labbiases);
                
        negdata = double(gt(negvisprobs,rand(numcases,numvis)));
        neglabs = sampling_y(neglabprobs, numcases, numlab);
        
        neghidprobs = h_given_xy(negdata, neglabs, vishid, labhid, hidbias, 1);
        
        %%% END OF NEGATIVE PHASE %%%

        %% Generative gradient (positive gradient - negative gradient) %%
        grad_vishid_gen = (poshidprobs * data'  - neghidprobs * negvisprobs') / numcases;
        grad_labhid_gen = (poshidprobs * label' - neghidprobs * neglabprobs') / numcases;
        grad_b_x_gen = (sum(data)  - sum(negvisprobs)) / numcases;
        grad_b_y_gen = (sum(label) - sum(neglabprobs)) / numcases;
        grad_b_h_gen = (sum(poshidprobs) - sum(neghidprobs)) / numcases;
                         
        posprods  = data' * poshidprobs;
        negprods  = negdata'*neghidprobs;
        neghidact = sum(neghidprobs);
        negvisact = sum(negdata); 
        poshidact = sum(poshidprobs);
        
        %% Discriminative Phase %%
        %initialization of the gradients
        g_b_h_disc_acc = 0;
        g_vishid_disc_acc = 0;
        g_labhid_disc_acc = zeros(numlab, numhid); 
        
        %ausiliary variables
        hidden_term = data * vishid;

        o_t_y = zeros(numlab, numcases, numhid,'double','gpuArray');
        neg_free_energ = zeros(numcases, numlab,'double','gpuArray');
        for yy = 1:numlab
            o_t_y(yy,:,:) = bsxfun(@plus, hidbiases + labhid(yy,:), hidden_term);
            neg_free_energ(:,yy) = labbias(yy) + sum(log(1 + exp(o_t_y(yy,:,:))),2);
        end;

        sum_tot_free_energ = sum_tot_free_energ + sum(sum(-label .* neg_free_energ));
        
        %we subtract its mean to the free energy in order to keep its values
        %smaller as possible, otherwise the exponetial in p(y|x) gets only inf
        %values.
        mean_free_energy = mean2(neg_free_energ);      
        p_y_given_x = exp(neg_free_energ - mean_free_energy);
        p_y_given_x = bsxfun(@rdivide, p_y_given_x, sum(p_y_given_x,2));

        dif_y_p_y = y - p_y_given_x;

        %update the gradient of the bias of y for the class 'iClass'
        g_b_y_disc_acc = sum(dif_y_p_y,2);

        %looping over classes is more efficient
        for iClass= 1:num_class
            sig_o_t_y = sig(o_t_y(:,:,iClass)); 

            sig_o_t_y_selected = sig_o_t_y(:,y(iClass,:)); 

            %update the gradient of the bias of h for the class 'iClass'
            g_b_h_disc_acc = g_b_h_disc_acc + sum(sig_o_t_y_selected, 2) - ...
            sum(bsxfun(@times, sig_o_t_y, p_y_given_x(iClass,:)),2);

            %update the gradient of w fot the class 'iClass'
            g_w_disc_acc = g_w_disc_acc + sig_o_t_y_selected * x(:,y(iClass,:))' - ...
            bsxfun(@times, sig_o_t_y, p_y_given_x(iClass,:)) * x';

            %update the gradient of u fot the class 'iClass'
            g_u_disc_acc(:,iClass) = sum(bsxfun(@times, sig_o_t_y, dif_y_p_y(iClass,:)), 2);
        end;

        %NOTE: the bias of x are not updated in the discriminative fase.

          g_b_y_disc = g_b_y_disc_acc / batch_size;
          g_b_h_disc = g_b_h_disc_acc / batch_size;
          g_w_disc = g_w_disc_acc / batch_size;
          g_u_disc = g_u_disc_acc / batch_size;
        
        %% Calculate error
        err = sum(sum((data - negdata).^2 ));
        errsum = err + errsum;

        %% Change momentum value
        if epoch > 5
            momentum = finalmomentum;
        else
            momentum = initialmomentum;
        end

        %% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        vishidinc = momentum * vishidinc + ...
                    epsilonw * ( grad_vishid_gen - weightcost*vishid);
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
    wait(gpuDevice)
    
    if mod(epoch,5) == 0
        fprintf('Saving %d-st layer data to a file..\n', layer_number);
        save(sprintf('%s/DRBM_L%d_%d_%d.mat', data_path, layer_number, numvis, numhid),...
            'vishid','visbiases','hidbiases','epoch','err_hist');
    end
end

wait(gpuDevice)
fprintf('Saving %d-st layer data to a file..\n', layer_number);
save(sprintf('%s/DRBM_L%d_%d_%d.mat', data_path, layer_number, numvis, numhid),...
    'vishid','visbiases','hidbiases','epoch','errsum','err_hist');
