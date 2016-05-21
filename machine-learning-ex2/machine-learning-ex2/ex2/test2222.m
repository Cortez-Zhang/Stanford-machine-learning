clear ; close all; clc
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
%Part 1: Regularized Logistic Regression 

% Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));
% Initialize fitting parameters
theta = zeros(size(X, 2), 1);
% Set regularization parameter lambda to 1
lambda = 1;
% Compute and display initial cost and gradient for regularized logistic
% regression


m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

M=sigmoid(X*theta);
M1=log(M);
M2=log(1-M);
% M=[M1,M2];
N1=-y;
N2=-(1-y);
% N=[N1;N2];
J=(1/m)*sum(N1.*M1+N2.*M2)+lambda/(2*m)*(sum(theta.^2)-theta(1,1)^2);
%Gradient;
rio=[0;ones((length(theta)-1),1)];
grad=(1/m)*X'*(M-y)+(lambda/m)*theta.*rio;


