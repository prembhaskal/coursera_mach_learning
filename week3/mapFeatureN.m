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

function [out] = mapFeatureN (X1, X2, N)
% MAPFEATUREN feature mapping function to polynomial features
% 
% It takes 3 inputs - X1, X2 and N
% returns out matrix with feature values comprising of different degress
% different feature values are return all combinations of m and n X1^m . X2^n 
% such that m + n <=N, m,n > 0.

out = ones(size(X1,1),0);
for deg=0:N
  for m=0:deg
    out(:,end+1) = (X1 .^ m) .* (X2 .^ (deg-m));
   end
end

endfunction
