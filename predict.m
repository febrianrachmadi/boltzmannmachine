function [ error ] = predict( x, y, W, U, b_x, b_h, b_y )
%PREDICT Summary of this function goes here
%   Detailed explanation goes here

    numcases = size(x, 1);
    numlab = size(y, 2);
    hidden_term = x * W;
    neg_free_energ = zeros(numcases, numlab,'double','gpuArray');
    for yy = 1:numlab
        neg_free_energ(:,yy) = b_y(yy) + sum(log(1 + ...
            exp(bsxfun(@plus, b_h + U(yy,:), hidden_term ))),2);
    end;
    [~, prediction] = max(neg_free_energ');
        
    sum_err = 0;
    for nn = 1 : numcases
       sum_err = sum_err + ne(y(nn, prediction(nn)),1); 
    end
    
    error = sum_err / numcases * 100;
end

