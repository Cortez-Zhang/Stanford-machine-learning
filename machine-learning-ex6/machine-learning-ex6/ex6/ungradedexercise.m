%2.6 ungraded exercise.

close all;clear ;clc;
% the non spam data
%% get the train data;
addpath ./easy_ham; %relative direct
file=dir('./easy_ham\*.*');
for n=3:length(file)
    file_contents{n-2,:}=readFile(file(n).name);
    features(:,n-2) =emailFeatures(processEmail_self(file_contents{n-2,:}));%change the normal processEmail to unprint the contents.
end
% save features.mat; % comment in case to re run this tedious process
load('features.mat');

train_self_nonspam= features';
train_self_nonspamlable=zeros(size(train_self_nonspam,1),1);

train_X0=train_self_nonspam;
train_y0=train_self_nonspamlable;
% the spam data

addpath ./spam_2; %relative direct
file=dir('./spam_2\*.*');
for n=3:length(file)
    file_contents{n-2,:}=readFile(file(n).name);
    features_spam(:,n-2) =emailFeatures(processEmail_self(file_contents{n-2,:}));%change the normal processEmail to unprint the contents.
end
% save features_spam.mat;
load('features_spam.mat');

train_self_spam= features_spam';
train_self_spamlable=ones(size(train_self_spam,1),1);

train_X1=train_self_spam;
train_y1=train_self_spamlable;

train_X=[train_X0;train_X1];
train_y=[train_y0;train_y1];

%% random the data sheet.
m=size(train_X,1);
index=randperm(m);      %Generate the random index
train_X=train_X(index,:); 
train_y=train_y(index,:);      %Generate the random 
% we use the whole hard_ham as the test set.
save train_X;
save train_y;
%% train svm and predict

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(train_X, train_y, C, @linearKernel);

p = svmPredict(model, train_X);

fprintf('Training Accuracy: %f\n', mean(double(p == train_y)) * 100);

 % test  data
addpath ./hard_ham; %relative direct
file=dir('./hard_ham\*.*');
for n=3:length(file)
    file_contents{n-2,:}=readFile(file(n).name);
    features_test(:,n-2) =emailFeatures(processEmail_self(file_contents{n-2,:}));%change the normal processEmail to unprint the contents.
end
% save features_test.mat; % comment in case to re run this tedious process
load('features_test.mat');

test_self_nonspam= features_test';
test_self_nonspamlable=zeros(size(test_self_nonspam,1),1);

test_X=test_self_nonspam;
test_y=test_self_nonspamlable;

% predect

p1 = svmPredict(model, test_X);

fprintf('Test Accuracy: %f\n', mean(double(p1 == test_y)) * 100);

%since the test accuracy is 26.8, so the test_X is signifigant diffenent from the train set.

%% change the data stracure.
close all;clear ;clc;
% nonspam;
load('features.mat');
load('features_test.mat');

train_self_nonspam2= features';
train_self_nonspam1= features_test';

train_self_nonspam=[train_self_nonspam1;train_self_nonspam2];
train_self_nonspamlab=zeros(size(train_self_nonspam,1),1);

train_X0=train_self_nonspam;
train_y0=train_self_nonspamlab;

% spam 

load('features_spam.mat');

train_self_spam= features_spam';
train_self_spamlable=ones(size(train_self_spam,1),1);

train_X1=train_self_spam;
train_y1=train_self_spamlable;

%merge
train_X=[train_X0;train_X1];
train_y=[train_y0;train_y1];

%random
m=size(train_X,1);
index=randperm(m);      %Generate the random index
train_X=train_X(index,:); 
train_y=train_y(index,:);      %Generate the random 

% choose the train set and the test set;

train_XX=train_X(1:4000,:);
train_yy=train_y(1:4000,:);

test_XX=train_X(4001:end,:);
test_yy=train_y(4001:end,:);

%train and predict

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(train_XX, train_yy, C, @linearKernel);

p = svmPredict(model, train_XX);

fprintf('Training Accuracy: %f\n', mean(double(p == train_yy)) * 100);


p1 = svmPredict(model, test_XX);

fprintf('Test Accuracy: %f\n', mean(double(p1 == test_yy)) * 100);
