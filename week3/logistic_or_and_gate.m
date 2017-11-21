## Copyright (C) 2017 bhaskal
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} logistic_or_gate (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: bhaskal <bhaskal@5CG4340QNZ>
## Created: 2017-11-21

function logistic_or_and_gate ()
  
  fprintf("calculating theta values for OR gate\n\n");
  
  X = [0 0; 0 1; 1 0; 1 1;];
  y = X(:,1) | X(:,2); %[0; 1; 1; 1;];
  X = [ones(size(X,1),1) X];
  theta = zeros(size(X,2),1);
  
  % optimize using fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  costFunctionTheta = @(theta)(costFunction(theta, X, y));
  [theta, cost] = fminunc(costFunctionTheta, theta, options);
  
  fprintf('cost from fminunc is %f\n', cost);
  fprintf('theta from fminunc is \n');
  fprintf(' %f\n', theta);
  fprintf("********************************************\n\n");
  
  fprintf('printing hypothesis function for or gate values\n');
  fprintf('%d %d %1.2f\n',[X(:,[2:3]), sigmoid(X * theta)]'); % this prints values side by side.
  fprintf("calculating theta values for AND gate\n\n");
  
  X = [0 0; 0 1; 1 0; 1 1;];
  y = X(:,1) & X(:,2);
  X = [ones(size(X,1),1) X];
  theta = zeros(size(X,2),1);
  
  % optimize using fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  costFunctionTheta = @(theta)(costFunction(theta, X, y));
  [theta, cost] = fminunc(costFunctionTheta, theta, options);
  
  fprintf('cost from fminunc is %f\n', cost);
  fprintf('theta from fminunc is \n');
  fprintf(' %f\n', theta);
  
  fprintf('printing hypothesis function for AND gate values\n');
  fprintf('%d %d %1.2f\n',[X(:,[2:3]), sigmoid(X * theta)]');
  fprintf("********************************************\n\n");


  fprintf("calculating theta values for XNOR gate\n\n");
  fprintf("logistic regression with only 2 variable fails to find satisfactory answer with just 2 variables");
  
  X = [0 0; 0 1; 1 0; 1 1;];
  y = not(xor(X(:,1), X(:,2)));
  X = [ones(size(X,1),1) X];
  theta = zeros(size(X,2),1);
  
  % optimize using fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  costFunctionTheta = @(theta)(costFunction(theta, X, y));
  [theta, cost] = fminunc(costFunctionTheta, theta, options);
  
  fprintf('cost from fminunc is %f\n', cost);
  fprintf('theta from fminunc is \n');
  fprintf(' %f\n', theta);
  
  fprintf('hypothesis function of XNOR without feature mapping.\n');
  fprintf('%d %d %1.2f\n',[X(:,[2:3]), sigmoid(X * theta)]');
  
  fprintf("********************************************\n\n");
  
  
  fprintf("calculating theta values for XNOR gate with feature mapping to degree 2\n\n");
  
  X = [0 0; 0 1; 1 0; 1 1;];
  y = not(xor(X(:,1), X(:,2)));
  
  N = 2;
  X = mapFeatureN(X(:,1), X(:,2), N);
  fprintf('feature count after the mapping to degree %d is %d\n', N, size(X,2));
  theta = zeros(size(X,2),1);
    
  % optimize using fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  lambda = 0;
  costFunctionRegT = @(theta)(costFunctionReg(theta, X, y, lambda));
  [theta, cost] = fminunc(costFunctionRegT, theta, options);
  
  fprintf('cost from fminunc is %f\n', cost);
  fprintf('theta from fminunc is \n');
  fprintf(' %f\n', theta);
  
  fprintf('hypothesis function of XNOR with feature mapping.\n');
  fprintf('%d %d %1.2f\n',[X(:,[2:3]), sigmoid(X * theta)]');
  fprintf("********************************************\n\n");
  
  

endfunction
