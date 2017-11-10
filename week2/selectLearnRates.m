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

function [] = selectLearnRates()
%selectLearnRates plots the cost function using different learning rates
% it works on week2 data.
data = load('ex1data2.txt');
size(data)
n = size(data,2)
X = data(:,[1:n-1]); 
y = data(:,[n]);

printf('size of X is ');disp(size(X));
printf('size of Y is ');disp(size(y));

%printf('printing vales of the Y \n');
%disp(y(1:10));

X_Norm = featureNormalize(X);
printf('size of X_Norm is ');disp(size(X_Norm));
% check data after normalization
%X_Norm([1:10],:)
% append ones
X_Norm = [ones(size(X_Norm,1),1) X_Norm];

Y_Norm = featureNormalize(y);
%printf('printing some data after normalizing y');
%disp(Y_Norm([1:3],1));

% check again
printf('size of X_Norm is ');disp(size(X_Norm));
X_Norm([1:3],:)

num_iters = 50;
alpha = 0.1;
theta = zeros(n,1) % theta values start from theta0, theta1, theta2 ... thetaN
printf('size of theta is ');disp(size(theta));
[theta1, J_Hist] = gradientDescentMulti(X_Norm, Y_Norm, theta, alpha, num_iters);
%printf('size of J_Hist is ');disp(size(J_Hist));
plot(1:50, J_Hist(1:50), 'b');
xlabel('no of iterations');
ylabel('cost function value');

end