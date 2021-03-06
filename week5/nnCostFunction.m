function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X_orig = X;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

  % converting the output to K class outputs, each output is vector of K labels.
  ky = zeros(m,num_labels); % size [m,num_labels]
  for i=1:m
    ky(i,y(i,1)) = 1;
  end
  
  
  % calculating h_theta(x)  = a(3) --> with 1 hidden layer.
  X = [ones(size(X,1),1) X];
  %disp('size of X is \n'); disp(size(X));
  z_2 = Theta1 * X';  %size [S(2), S(1)+1] * [S(1)+1,m] = [S(2),m]
  a_2 = sigmoid(z_2); % size [S(2),m]
  a_2 = a_2';  %size [m,S(2)]
  
  a_2 = [ones(size(a_2,1),1) a_2]; %size [m,S(2)+1]
  %disp('size of a_2 is \n');disp(size(a_2));
  z_3 = Theta2 * a_2'; %size [S(3),S(2)+1] * [S(2)+1,m] = [S(3),m] = [num_labels,m]
  a_3 = sigmoid(z_3);% size [S(3),m]
  a_3 = a_3'; %size [m,S(3)] = [m,num_labels]
  
  h_theta_x = a_3; %size[m,num_labels] -- each row is k class vector, one row each for each training set.
  


  

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


  cost = 0;
  for i=1:m
    % term1 = y(i) * log(h_theta_x) --> for each training set, multiply k class entries (one by one)    
    % y is stored rowwise, one row per training set with k columns
    % h_theta_x is the same size as y
    term1 = ky(i,:) * log(h_theta_x(i,:))'; %constant value
    term2 = (1 .- ky(i,:)) * log(1 .- h_theta_x(i,:))';  %constant value
    cost = cost - (term1 + term2)/m;
  end
  
  J = cost;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

  % iterate over training set
  DEL_2 = zeros(size(Theta2));
  DEL_1 = zeros(size(Theta1));
  % for i=1:m
    
    % % fwd propagation for layer 1
    % X_i = X_orig(i,:); % get ith row from input
    % X_i = X_i'; % size [s(1),1]
    % X_i = [1;X_i]; % add bias unit - size[S(1)+1,1]
    % Theta1;%size[S(2), S(1)+1]
    % z_2 = Theta1 * X_i;%size[S(2),S(1)+1] * [S(1)+1,1] = [S(2),1]
    % a_2 = sigmoid(z_2);%size[S(2),1]
    
    % % fwd propagation for layer 2
    % a_2 = [1; a_2]; %size[S(2)+1,1]
    % Theta2;%size[K,S(2)+1] K = num_labels
    % z_3 = Theta2 * a_2;%size[K,1]
    % a_3 = sigmoid(z_3); %size[K,1]
    
    % % back propagation - del3
    % y_i = zeros(num_labels,1); % size[K,1]
    % y_i(y(i,1),1) = 1;
    % del_3 = a_3 - y_i; %size[K,1]
    
    % % back propagtion - del2
    % % del_2 = theta2
    % del_2 = (Theta2' * del_3); %size[S(2)+1,K] * [K,1] = [S(2)+1,1]
    % % remove bias unit from del_2
    % del_2 = del_2(2:end,1) .* sigmoidGradient(z_2); %size [S(2),1] 
    
    % % accumulate in DEL2
    % DEL_2 = DEL_2 .+ (del_3 * a_2'); % size [K,1] * [1, S(2)+1]     = [K,S(2)+1] 
    
    % % accumulate in DEL1
    % DEL_1 = DEL_1 .+ (del_2 * X_i');% size[S(2),1] * [1, S(1)+1] = [S(2), S(1)+1]
  % end
  
  % fwd propagation without loop.
  X_orig; %size[m,S1]
  X_n = X_orig'; %[S1,m] -- column vector for each training set
  X_n = [ones(1,m); X_n]; %[S1+1,m]
  Theta1; %[S2, S1 + 1]
  z_2 = Theta1 * X_n  ;%[S2,m]
  a_2 = sigmoid(z_2);%[S2,m]
  
  a_2 = [ones(1,m); a_2];%[S2+1,m]
  Theta2;%[k,S2+1]
  z_3 = Theta2 * a_2;%[k,m]
  a_3 = sigmoid(z_3); %[k,m]
  
  % back propagation without loop
  y_n = zeros(num_labels,m);%[k,m]
  for i=1:m
	y_n(y(i,1),i) = 1;
  end
  
  del_3 = a_3 .- y_n ;%[k,m]
  del_2 = (Theta2'(2:end,:) * del_3) .* sigmoidGradient(z_2); % same size as a2 without bias unit --> [S2,m]
  
  DEL_2 = del_3 * a_2';% [k,m] * [m,S2+1] =[k,S2+1]
  DEL_1 = del_2 * X_n';% [S2,m] * [m,S1+1] =[S2,S1+1]
  
  Theta1_grad = DEL_1 ./ m;
  Theta2_grad = DEL_2 ./ m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% cost regularization without bias unit
% reg_cost = lambda/2m * ( theta1^2 + theta2^2);
  regT1 = Theta1(:,2:end);
  regT1Vector = regT1(:);
  rterm1 = regT1Vector' * regT1Vector;
  
  regT2 = Theta2(:,2:end);
  regT2Vector = regT2(:);
  rterm2 = regT2Vector' * regT2Vector;
  
  reg_cost = lambda/(2 * m) * (rterm1 + rterm2);
  
  J = cost + reg_cost;
  
  
  % reg_cost for the gradient calculation
  % grad = 1/m * DEL + lambda/m*Theta(i,j)  where j >0
  
  
  Theta1_reg = Theta1;
  Theta1_reg(:,1) = zeros(size(Theta1,1),1);
  grad1_reg = (Theta1_reg .* lambda) ./ m;
  Theta1_grad = Theta1_grad .+ grad1_reg;
  
  Theta2_reg = Theta2;
  Theta2_reg(:,1) = zeros(size(Theta2,1),1);
  grad2_reg = (Theta2_reg .* lambda) ./ m;
  Theta2_grad = Theta2_grad .+ grad2_reg;
  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
