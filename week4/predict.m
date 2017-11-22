function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

fprintf('size of Theta1\n');
fprintf(' %f\n', size(Theta1));

fprintf('size of Theta2\n');
fprintf(' %f\n', size(Theta2));

fprintf('size of X\n');
fprintf(' %f\n', size(X));

% appending bias units to the input - a(1)
X = [ones(size(X,1),1) X];

fprintf('size of X\n');
fprintf(' %f\n', size(X));

% z(2) = Theta1 * a(1)  and a(2) = g(z(2))
Z_2 = Theta1 * X'; % size[25, 5000] - each row represent data in that unit
Z_2 = Z_2'; % each row is for each training set. each column in for each unit in layer 2;
a2 = sigmoid(Z_2); % size[5000 * 25]

fprintf('size of a2 is \n');
fprintf(' %f\n', size(a2));

% z(3) = Theta2 * a(2)  and a(2) = g(z(3));
a2 = [ones(size(a2,1),1) a2]; % adding bias unit in each training set. extra feature that is
Z_3 = Theta2 * a2';
Z_3 = Z_3';
a3 = sigmoid(Z_3); % size[5000 * 10] % each row is for each training set, with each column represent 
% probability for that particular label, label 10 is 0.

% find which label has max probability for each training set.
% so the index is the actual probability.
[val, p] = max(a3,[],2);

% =========================================================================


end
