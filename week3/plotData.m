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

admit = find(y == 1); % indices where y = 1;
notAdmit = find(y == 0); % indicies where y = 0;

% plot not admitted in yellow
plot(X(notAdmit,1), X(notAdmit,2), 'ko', 'markerfacecolor', 'yellow', 'markersize', 7);
% plot admitted in small black cross
plot(X(admit,1), X(admit,2), 'k+', 'markersize', 7, 'linewidth', 2);


% =========================================================================



hold off;

end
