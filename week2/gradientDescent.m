function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% printing the values of matrices to be sure
size(theta) % n X 1
size(X)     % m X n
size(y)     % m X 1
alpha

  for iter = 1:num_iters

      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta. 
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCost) and gradient here.
      %

      % theta is a matrix so don't assume its size as fixed value in code.

      % implementing the gradient function step by step here
      %interim = theta' * X'; % interim size is 1 X m
      %interim = interim' - y; % interim size is m X 1
      %interim = interim'; % 1 X m 
      %interim = interim * X; % 1 X n, this matrix multiplication takes care of summation across m training sets
      %interim = interim'; % n X 1 that is 1 value per column
      %theta = theta - (alpha/m) .* interim;
      
      % another way to do this
      %interim = X * theta - y; % size [m x 1]
      %interim = interim' * X; % size [1 x m] * [m x n] = [1 * n]
      %interim = interim';
      interim = ((X * theta - y)' * X)';
      theta = theta .- (alpha/m) .* interim;

      % ============================================================

      % Save the cost J in every iteration    
      J_history(iter) = computeCost(X, y, theta);
      %J_history(iter)  % uncomment to print the cost function value after every iteration

  end

end
