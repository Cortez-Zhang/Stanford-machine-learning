function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

index1=find(y);
index2=find(y-1);
plot(X(index1,1),X(index1,2),'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(index2,1),X(index2,2),'ko','MarkerFaceColor', 'y', ...
'MarkerSize', 7);

%------------------------reference---------------------------------
% % Find Indices of Positive and Negative Examples
% pos = find(y==1); neg = find(y == 0);
% % Plot Examples
% plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
% 'MarkerSize', 7);
% plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
% 'MarkerSize', 7);
%------------------------reference---------------------------------




% =========================================================================



hold off;

end
