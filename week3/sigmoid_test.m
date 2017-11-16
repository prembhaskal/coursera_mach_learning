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
## @deftypefn {} {@var{retval} =} sigmoid_test (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: bhaskal <bhaskal@5CG4340QNZ>
## Created: 2017-11-16

function [] = sigmoid_test ()
  
  %figure;
  %hold on;
  
  test_z = zeros(20,1);
  length = size(test_z,1);
  
  % fill negative numbers
  for i=1:length/2
    test_z(i) = -1 * (length/2 - i);
  end
  
  test_z(length/2+1) = 0;
  
  % fill positive numbers;
  for i=length/2+2:length
    test_z(i) = (i - length/2);
  end
  
  %printf('displaying the test_z values %f\n', test_z);
  
  g = sigmoid(test_z);
  %printf('size of sigmoid value is');disp(size(g));
  
  %printf('displaying the sigmoid values %f\n', g);
  
  plot(test_z, g, 'x', 'linewidth', 2);
  %xlabel('z -->');
  %ylabel('sigmoid value -->');
  
  %hold off;

endfunction
