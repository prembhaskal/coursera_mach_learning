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
  fprintf("********************************************\n\n");

endfunction
