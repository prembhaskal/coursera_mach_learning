function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


% size of X is m x n
% size of theta is n x 1
thetaX = X * theta; % size is m x 1
hThetaX = sigmoid(thetaX); % size is m x 1
JTerm1 = y' * log(hThetaX); % constant value
JTerm2 = (1 .- y)' * log(1 .- hThetaX); % constant value

n = size(theta,1); % no. of rows
%printf('no. of features after featuremaping is %d\n', n);
thetaReg = theta([2:n],:);

JRegTerm = lambda/(2 * m) * (thetaReg' * thetaReg);  % A'A gives summation(A(i)^2)

J = -1/m * (JTerm1 + JTerm2) + JRegTerm;


% gradient
% for 1st term, theta 1
X_1 = X(:,1);
%disp('size of X_1 is ');disp(size(X_1));
grad(1,1) = 1/m * (X_1' * (hThetaX - y));

X_rem = X(:,[2:n]);
%disp('size of X_rem is ');disp(size(X_rem)); % size is [m x n-1]

grad([2:n],1) = 1/m .* (X_rem' * (hThetaX - y)) .+ (lambda/m) .* thetaReg;

% =============================================================

end
