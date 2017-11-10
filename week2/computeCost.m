function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% m - no. of traning set, n - no. of features
% theta  is n * 1
% X size is m * n
% y size is m * 1

%delta1 = theta' * X';
% delta size will be 1 * m
% delta2 = (theta' * X')' - y;
% delta2 will be m * 1
% delta3 = ((theta' * X')' - y) .^ 2;
% delta4 = sum(delta3);
% J = delta4 / (2 * m);
% J= 1/(2*m) * sum(((theta' * X')' - y) .^ 2);

% better way
% consider a = [a1 a2 ... am] then matrix equivalent of sum_i(a(i)^2) 
% is a * a'

interim = X*theta - y; % interim is of [m * 1]
J = (1/(2*m)) .* (interim' * interim);

% =========================================================================

end
