% Compute Transformation matrix
% t: difference in width/height between the reference points
% scale/sheer: x/x', y/y'
% theta: Maybe hardcode to check values?
% Add weight dimension
% Check week 7 lecture for matrix transformation example
% 8 rows

% im_right: A, B, C, D
% im_left: Ap, Bp, Cp, Dp


% Use it on the 4 corners of 1 image to find its relative position
% to the other image

% Part 3: Week 4: Keypoints

clear all;

im_left = imread('bikepath_left.jpg');
im_right = imread('bikepath_right.jpg');

im_left = double(im_left);
im_left = im_left(:, :, :) / 255.0;
im_right = double(im_right);
im_right = im_right(:, :, :) / 255.0;

A = left_cor(1);
B = left_cor(2);
C = left_cor(3);
D = left_cor(4);

right_cor_relative = zeros(3, 4);

for i=1:size(left_cor)
  right_cor_relative(i) = left_cor(i) - right_cor(i);
  % right_cor_relative(i) = [right_cor_relative(i) 1];
end

Ap = [right_cor_relative(1) 1];
Bp = [right_cor_relative(2) 1];
Cp = [right_cor_relative(3) 1];
Dp = [right_cor_relative(4) 1];

Am = zeros(8, 8);
Am(1, :) = [A(1) A(2) 1 0 0 0];
Am(2, :) = [0 0 A(1) A(2) 1 0];
Am(3, :) = [B(1) B(2) 1 0 0 0];
Am(4, :) = [0 0 0 B(1) B(2) 1];
Am(5, :) = [C(1) C(2) 1 0 0 0];
Am(6, :) = [0 0 0 C(1) C(2) 1];
Am(7, :) = [D(1) D(2) 1 0 0 0];
Am(8, :) = [0 0 1 D(1) D(2) 1];

b = [Ap(1); Ap(2); Bp(1); Bp(2); Cp(1), Cp(2); D(1), Dp(2)];
m = inv(Am' * Am) * Am' * b;

% Not sure if this is right
Mf = [m(1), m(2), m(3) m(4); m(5), m(6), m(7), m(8); 0 0 0 1];
% Check here:
% B = inv(Mf) * Bp;

