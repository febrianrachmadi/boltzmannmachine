function [samples] = sampling_y(probs, numcases, numlab)

    xx  = cumsum(probs,2);
    xx1 = rand(numcases,1);
    samples = zeros(numcases, numlab, 'double', 'gpuArray');
    for jj = 1:numcases
        index = find(xx1(jj) <= xx(jj,:), 1 );
        samples(jj,index) = 1;
    end

end