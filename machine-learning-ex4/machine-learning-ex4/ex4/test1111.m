clear ; close all; clc
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10; 


load('ex4data1.mat');
m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

lambda = 0;



Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

    X=[ones(m,1),X];%add x0 for every x;    
%     transform y to vector
    for i1=1:m
         y1=zeros(num_labels,1);% set a vector equals label number.
         y1(y(i1),1)=1;
         y2(i1,:)=y1;
    end
    y=y2;
%  %for loop verion.  
%   for t=1:m
%         %step1
%          a1=X(t,:);     % tth example
%          z2=a1*Theta1'; %weight value 
%          a2=sigmoid(z2);%activation
%          a2=[ones(1,1),a2];%add a20=1
%          z3=a2*Theta2'; %weight value 
%          a3=sigmoid(z3);%activation        
%          %step2
%          delta3=a3-y(t,:);
%          %step3
%          delta2=(delta3*Theta2(:,(2:end))).*(sigmoidGradient(z2));%this section
%                                                                   %didn't
%                                                                   %count
%                                                                   %the
%                                                                   %delta0
%          %step4
% %        delta2=delta2(2:end);
%          
%          D2=0;
%          D1=0;
%          D2=D2+delta3'*a2;
%          D1=D1+delta2'*a1;
%          
%     end
%     %obtain the unregularized gradient
%    Theta2_grad=1/m*D2;
%    Theta2_grad=1/m*D1;
% try not use for loop    
%  %for loop verion.  
          
%step1
         a1=X;     % tth example
         z2=a1*Theta1'; %weight value 
         a2=sigmoid(z2);%activation
         a2=[ones(m,1),a2];%add a20=1
         z3=a2*Theta2'; %weight value 
         a3=sigmoid(z3);%activation        
         %step2
         delta3=a3-y;
         %step3
         delta2=(delta3*Theta2(:,(2:end))).*(sigmoidGradient(z2));%this section
                                                                  %didn't
                                                                  %count
                                                                  %the
                                                                  %delta0
         %step4
%        delta2=delta2(2:end);
         
         D2=0;
         D1=0;
         D2=D2+delta3'*a2;
         D1=D1+delta2'*a1;
         
   
    %obtain the unregularized gradient
   Theta2_grad=1/m*D2;
   Theta1_grad=1/m*D1;
   grad = [Theta1_grad(:) ; Theta2_grad(:)];
   
 %% test the maxtier and lambda for diffent value.  
clear ; close all; clc


input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

load('ex4data1.mat');
m = size(X, 1);
load('ex4weights.mat');
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options1 = optimset('MaxIter', 500);
options2 = optimset('MaxIter', 100);
options3 = optimset('MaxIter', 300);
%  You should also try different values of lambda
lambda1 = 0.1;
lambda2 = 1;
lambda3 = 3;

% Create "short hand" for the cost function to be minimized
costFunction1 = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda1); 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params1, cost1] = fmincg(costFunction1, initial_nn_params, options1);

% Obtain Theta1 and Theta2 back from nn_params
Theta11 = reshape(nn_params1(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta21 = reshape(nn_params1((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('\nVisualizing Neural Network... \n')
figure;
displayData(Theta11(:, 2:end));

pred1 = predict(Theta11, Theta21, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred1 == y)) * 100);            
             
  
             
             
             
             
costFunction2 = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda2); 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params2, cost2] = fmincg(costFunction2, initial_nn_params, options2);

% Obtain Theta1 and Theta2 back from nn_params
Theta12 = reshape(nn_params2(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta22 = reshape(nn_params2((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
fprintf('\nVisualizing Neural Network... \n')
figure;
displayData(Theta12(:, 2:end));

pred2 = predict(Theta12, Theta22, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred2 == y)) * 100); 
             
             
             
             
             
             
costFunction3 = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda3); 
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params3, cost3] = fmincg(costFunction3, initial_nn_params, options3);

% Obtain Theta1 and Theta2 back from nn_params
Theta13 = reshape(nn_params3(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta23 = reshape(nn_params3((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



fprintf('\nVisualizing Neural Network... \n')
figure;
displayData(Theta13(:, 2:end));

pred3 = predict(Theta13, Theta23, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred3 == y)) * 100);

%%  if subtract the theta11 theta12 theta13 and display them again.
Theta1111=Theta11-Theta12;
figure;
displayData(Theta1111(:, 2:end));

Theta2222=Theta12-Theta13;
figure;
displayData(Theta2222(:, 2:end));

Theta3333=Theta11-Theta13;
figure;
displayData(Theta3333(:, 2:end));




