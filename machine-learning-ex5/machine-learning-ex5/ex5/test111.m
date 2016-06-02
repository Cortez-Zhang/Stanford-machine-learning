%test 
X=rand(10,1);
mu = mean(X);
X_norm = bsxfun(@minus, X, mu); %awesome tools

sigma1 = std(X_norm);
sigma2 = std(X);
X_norm = bsxfun(@rdivide, X_norm, sigma);
%% test the pre-progress
clear ; close all; clc
tic;
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);


p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones



lambda = 0.01;
iter=2;               %50 times random 
for i=1:iter
%first random choose

index=randperm(m);      %Generate the random index
X_poly=X_poly(index,:); 
y=y(index,:);           %Generate the random X_poly and y.

% [theta] = trainLinearReg(X_poly, y, lambda);
[error_train_temp(:,i), error_val_temp(:,i)] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
% error_train(:,i)=error_train_temp;
% error_val(:,i)=error_val_temp;
end
%then calculate the mean error_val and error_train
error_train=sum(error_train_temp,2)/iter;
error_val=sum(error_val_temp,2)/iter;

plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
T = toc;


