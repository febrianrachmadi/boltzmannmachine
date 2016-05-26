function [probs] = h_given_xy(x, y, W, U, b_h, end_layer)

    probs = 1./(1 + exp(-(x*end_layer*W) - (y*U) - b_h)); 
end