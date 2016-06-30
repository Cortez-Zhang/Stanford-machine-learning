function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = mean(X);
mu=mu';
% sigma = std(X); % Need normalized by N
sigma = std(X,1);
% S = std(A,w);
% When w = 0 (default), S is normalized by N-1. 
% When w = 1, S is normalized by the number of observations,
% N. w also can be a weight vector containing nonnegative elements. 
sigma2=(sigma.^2)';




% mu = mean(X);
% X_norm = bsxfun(@minus, X, mu);
% 
% sigma = std(X_norm);
% X_norm = bsxfun(@rdivide, X_norm, sigma);









% =============================================================


end
