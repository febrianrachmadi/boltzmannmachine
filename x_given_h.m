function [probs] = x_given_h(x, W, b_x)

    probs = 1./(1 + exp(-(x * W') - b_x)); 
end