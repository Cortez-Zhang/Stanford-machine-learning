clear ; close all; clc
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X);
% Add intercept term to x and X_test
X = [ones(m, 1) X];
% Initialize fitting parameters
theta = zeros(n + 1, 1);

m = length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
M=sigmoid(X*theta);
M1=log(M);
M2=log(1-M);
% M=[M1,M2];
N1=-y;
N2=-(1-y);
% N=[N1;N2];
J=(1/m)*sum(N1.*M1+N2.*M2);

%Gradient;

theta=(1/m)*X'*(M-y);


