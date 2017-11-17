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

function plotContourBoundayMine (theta, X, y, N)
% contour boundary, also needs the degree of feature mapping N along with other data.
    disp('printing boundary using custom code');
  % Plot Data using our function, skip the first column which is all ones.
    plotData(X(:,2:3), y);
    hold on;
    
    % creata a grid of points to help us plot the decision boundary.
    u = linspace(30, 100, 100);
    v = linspace(30,100, 100);
    
    % decision boundary is the curve theta' * X = theta0 . X0 + theta1. X1 + ... + thetaN.XN = 0
    % after feature mapping, we have many features to fit data more closely
    % here u and v are like X1 and X2, we need to convert them using feature mapping to

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            % mapFeature(val1, val2) will return [1 x n] vector. the below multiplication calculates 
            % the value for theta'.X for us.
            z(i,j) = mapFeatureN(u(i), v(j),N)*theta;
        end
    end
    
   % http://in.mathworks.com/help/matlab/ref/contour.html check this for why transpose
   % it is kind of design decision of transpose that below conditions are met
   % length(X) must equal size(Z,2) AND length(Y) must equal size(Z,1) when X,Y are vectors
   
   z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    % contour(u, v, z, [0, 0], 'LineWidth', 2);
    contour(u, v, z, [0.5, 0.5], 'LineWidth', 2);
    % removing [0,0] prints lot of contours in the image.
    %contour(u, v, z, 'LineWidth', 2);
    
    hold off;
endfunction
