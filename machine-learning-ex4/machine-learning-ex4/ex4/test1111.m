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
   
    


