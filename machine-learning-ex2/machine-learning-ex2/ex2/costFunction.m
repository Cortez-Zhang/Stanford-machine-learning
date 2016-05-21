function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
M=sigmoid(X*theta);
M1=log(M);
M2=log(1-M);
% M=[M1,M2];
N1=-y;
N2=-(1-y);
% N=[N1;N2];
J=(1/m)*sum(N1.*M1+N2.*M2);

%Gradient;

grad=(1/m)*X'*(M-y);


% =============================================================

end
