function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% computing cost function with regularized param

hthetaX = sigmoid(X * theta); % X[m*n] * theta[n*1]
JTerm1 = y' * log(hthetaX);% size m * 1
JTerm2 = (1 .- y)' * log(1 .- hthetaX);% [1,m] * [m,1]

% removing theta(0) so that it is nt
regTheta = theta; % size [n * 1]
regTheta(1,1) = 0;
JTermReg = lambda/(2 * m) * (regTheta' * regTheta);

J = (-1/m) * (JTerm1 + JTerm2) + JTermReg; % scalar value

% computing grad function
gradTerm1 = 1/m .* (X' * (hthetaX .- y)); % [nx1] vector
gradRegTerm = (lambda/m) .* regTheta; % [n x 1] vector

grad = gradTerm1 .+ gradRegTerm;

% =============================================================

grad = grad(:);

end
