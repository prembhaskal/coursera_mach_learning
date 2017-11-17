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
## @deftypefn {} {@var{retval} =} polynomial_function_q1 (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: bhaskal <bhaskal@5CG4340QNZ>
## Created: 2017-11-17

function polynomial_function_q1 ()
  
  %% Initialization
  clear ; close all; clc
  
  % load admission training set
  data = load('ex2data1.txt');
  
  fprintf('data loaded of size\n');
  fprintf(' %f\n', size(data));
  
  m = size(data,1);
  fprintf('no. of training set examples are %d\n', m);
  
  % feature and output y
  X = data(:,[1:2]);
  
  y = data(:,3);
  
  % plot data - showing positive and negative results
  plotData(X, y);
  
  X = [ones(m,1) X];
  
  disp(size(X));
  
  % features count = 3 (with 1 for theta0)
  % n = 3;
  
  theta = zeros(size(X,2),1);
  fprintf('displaying values of theta\n');
  fprintf(' %f\n',theta);
  
  % compute cost and grad for the theta values for initial theta.
  [J, grad] = costFunction(theta, X, y);
  
  fprintf('cost function value for initial theta is %f\n', J);
  fprintf('displaying gradient with initial theta\n');
  fprintf(' %f\n', grad);
  
  % optimize using fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  
  % calling fminunc with values fminunc (FCN, X0, OPTIONS)
  % see this for anonymous functions https://www.gnu.org/software/octave/doc/interpreter/Anonymous-Functions.html#Anonymous-Functions
  costFunctionTheta = @(theta)(costFunction(theta, X, y))
  [theta, cost] = fminunc(costFunctionTheta, theta, options);
  
  fprintf('cost from fminunc is %f\n', cost);
  fprintf('theta from fminunc is \n');
  fprintf(' %f\n', theta);
  
  % plot decision boundary theta'.X = 0
  plotDecisionBoundary(theta,X,y);
  
  hold on;
  xlabel('test score 1');
  ylabel('test score 2');
  hold off;
  
  % now try with more polynomial based features.
  X1 = data(:,1);
  X2 = data(:,2);
  N = 2;
  
  % map also appends columns of 1, so no X(0) features are needed.
  X = mapFeatureN(X1, X2, N);
  fprintf('feature count after the mapping to degree %d is %d\n', N, size(X,2));
  
  theta = zeros(size(X,2),1);
  
  % Set regularization parameter lambda 
  lambda = 1000;
  [J, grad] = costFunctionReg(theta, X, y, lambda);
  
  fprintf('after feature mapping\n');
  fprintf('cost is %f\n', J);
  %fprintf('grad is \n');fprintf(' %f\n', grad);
  
  % optimize using fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 400);
  % anonymous function for fminunc
  costFunctionRegT = @(theta)(costFunctionReg(theta, X, y, lambda));
 [theta, cost] = fminunc(costFunctionRegT, theta, options);
  
  fprintf('cost from fminunc is %f\n', cost);
  %fprintf('theta from fminunc is \n');
  %fprintf(' %f\n', theta);
  
  plotContourBoundayMine (theta, X, y, N);
  
endfunction
