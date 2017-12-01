function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%disp('size of X is'); disp(size(X));
%disp('size of theta is'); disp(size(theta));
%disp('size of y is'); disp(size(y));

term1 = X * theta - y;%[m,1]
half_sq_err_sum = (term1' * term1)/(2*m);

% reg term
thetaReg = theta;
thetaReg(1,1) = 0;
regTerm = lambda/(2*m) * (thetaReg'*thetaReg);

J = half_sq_err_sum + regTerm;


% grad calculation
gradTerm = X' * (X*theta - y); %[n,m] * [m,1] = [n,1]
grad = (1/m) .* (gradTerm .+ (lambda .* thetaReg)); % element wise mutiply and add

% =========================================================================

grad = grad(:);

end
