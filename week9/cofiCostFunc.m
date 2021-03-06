function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

  %  cost function for collaborative filter
%  term = X * Theta'- Y;
%  term = term .^ 2;
%  term = sum(sum(term .* R));
%  J = term / 2.0;
  
  % another way to do this. :)
  term = (X*Theta' - Y) .* R;
  J = (term(:)' * term(:))/2.0;
    
  Jreg = (lambda/2.0) * (X(:)' * X(:) + Theta(:)' * Theta(:));
  J = J + Jreg;

%  X_grad(i,k) = sum(X(i) * Theta(j) - y(i,j)) * Theta(j,k)
%  X_grad(i) = (X(i)*Theta(j) - y(i,j)) * Theta(J) % for all the features for a user
  X_grad = (( X* Theta' - Y) .* R ) * Theta;  % size [nm, nu] * [nu, n] = [nm, n]
  X_grad = X_grad + lambda .* X;
  
%  Theta_grad(j,k) = sum(X(i)*Theta(j) - Y(i,j) * X(i,k))

  Theta_grad = ((X * Theta' - Y) .* R)' * X; % size [nm,n] *[nm, nu]  = [nu, n]
  Theta_grad = Theta_grad + lambda .* Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
