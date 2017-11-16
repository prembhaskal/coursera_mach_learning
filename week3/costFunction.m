function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%disp('size of X is '); disp(size(X)); % size of X is [m x n]
%disp('size of y is '); disp(size(y)); % size of y is [m x 1]
%disp('size of theta is '); disp(size(theta)); % size of theta is [n x 1]


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% theta' * X
thetaX = X * theta; % results is data of [m x 1] size
hThetaX = sigmoid(thetaX); % retains size [m x 1]

term1 = y' * log(hThetaX); % constant
term2 = (1 .- y)' * log(1 .- hThetaX); % constant

J = -1 * (1/m) * (term1 + term2); % cost function formula

% gradient is given by
% 1/m * summation_1_m((H_theta_x(i) - y(i)) * xj(i) 
% = 1/m * [xj(0) * (H_theta_x(0) - y(0)) + ... + xj(m) * (H_theta_x(m) - y(m))]

grad = 1/m * (X' * (hThetaX - y)); % gradient formula -- partial_derivative(J(theta))

% =============================================================

end
