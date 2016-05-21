% test the regularization parameters influence.
%% Initialization
clear ; close all; clc
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda1 = 0;
lambda2 = 1;
lambda3 = 100;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost1, grad1] = costFunctionReg(initial_theta, X, y, lambda1);
[cost2, grad2] = costFunctionReg(initial_theta, X, y, lambda2);
[cost3, grad3] = costFunctionReg(initial_theta, X, y, lambda3);
fprintf('Cost at initial theta (zeros): %f\n', cost1);
fprintf('Cost at initial theta (zeros): %f\n', cost2);
fprintf('Cost at initial theta (zeros): %f\n', cost3);
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);%Due to the first fminunc
                                                    % exceeded the iteration limit,so change the first options            
options1 = optimset('GradObj', 'on', 'MaxIter', 1000);
% Optimize
[theta1, J1, exit_flag1] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda1)), initial_theta, options1);
[theta2, J2, exit_flag2] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda2)), initial_theta, options);
[theta3, J3, exit_flag3] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda3)), initial_theta, options);

% Plot Boundary1
plotDecisionBoundary(theta1, X, y);
hold on;
title(sprintf('lambda = %g', lambda1))
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
% Compute accuracy on our training set
print(gcf,'-dpng','lambda1.png') ;
p1= predict(theta1, X);
fprintf('Train Accuracy: %f\n', mean(double(p1 == y)) * 100);

% Plot Boundary2
plotDecisionBoundary(theta2, X, y);
hold on;
title(sprintf('lambda = %g', lambda2))
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
% Compute accuracy on our training set
print(gcf,'-dpng','lambda2.png') ;
p2= predict(theta2, X);
fprintf('Train Accuracy: %f\n', mean(double(p2 == y)) * 100);

% Plot Boundary3
plotDecisionBoundary(theta3, X, y);
hold on;
title(sprintf('lambda = %g', lambda3))
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
% Compute accuracy on our training set
print(gcf,'-dpng','lambda3.png') ;
p3= predict(theta3, X);
fprintf('Train Accuracy: %f\n', mean(double(p3 == y)) * 100);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% use the non-reg cost function to try.
clear ; close all; clc
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda1 = 0;
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost1, grad1] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter',1100);%Due to the first fminunc
                                                    % exceeded the iteration limit,so change the first options            

% Optimize
[theta1, J1, exit_flag1] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Plot Boundary1
plotDecisionBoundary(theta1, X, y);
hold on;
title(sprintf('lambda = %g', lambda1))
% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
% Compute accuracy on our training set
p1= predict(theta1, X);
fprintf('Train Accuracy: %f\n', mean(double(p1 == y)) * 100);

