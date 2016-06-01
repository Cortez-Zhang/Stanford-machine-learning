%test 
X=rand(10,1);
mu = mean(X);
X_norm = bsxfun(@minus, X, mu); %awesome tools

sigma1 = std(X_norm);
sigma2 = std(X);
X_norm = bsxfun(@rdivide, X_norm, sigma);