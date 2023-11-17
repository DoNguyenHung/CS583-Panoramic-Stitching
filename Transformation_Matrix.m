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
alpha = 0.3;

im_left = imread('bikepath_left.jpg');
im_right = imread('bikepath_right.jpg');
im_left = imresize(im_left, 0.25); 
im_right = imresize(im_right, 0.25);

im_left = double(im_left);
im_left = im_left(:, :, :) / 255.0;
im_right = double(im_right);
im_right = im_right(:, :, :) / 255.0;

width_right = size(im_right, 2);
height_right = size(im_right, 1);

% the resized locations left (Y,X)
branchlft= [245 950];
rocklft= [829 853];
signlft= [730 598];
leaflft= [1273 1044];

% the resized locations right (Y,X)
branchrgt= [319 606];
rockrgt= [850 511];
signrgt= [750 259];
leafrgt= [1258 628];

left_cor = [branchlft; rocklft; signlft; leaflft];
right_cor = [branchrgt; rockrgt; signrgt; leafrgt];

A = [left_cor(1, :) 1];
B = [left_cor(2, :) 1];
C = [left_cor(3, :) 1];
D = [left_cor(4, :) 1];

Ap = [right_cor(1, :) 1];
Bp = [right_cor(2, :) 1];
Cp = [right_cor(3, :) 1];
Dp = [right_cor(4, :) 1];

Am = zeros(8, 6);
Am(1, :) = [A(1) A(2) 1 0 0 0];
Am(2, :) = [0 0 0 A(1) A(2) 1];
Am(3, :) = [B(1) B(2) 1 0 0 0];
Am(4, :) = [0 0 0 B(1) B(2) 1];
Am(5, :) = [C(1) C(2) 1 0 0 0];
Am(6, :) = [0 0 0 C(1) C(2) 1];
Am(7, :) = [D(1) D(2) 1 0 0 0];
Am(8, :) = [0 0 0 D(1) D(2) 1];

b = [Ap(1); Ap(2); Bp(1); Bp(2); Cp(1); Cp(2); Dp(1); Dp(2)];
m = inv(Am' * Am) * Am' * b;

M = [m(1), m(2), m(3); m(4), m(5), m(6); 0 0 1];
% Check here:
% B = inv(M) * Bp';

corner1 = [1; 1; 1];
corner2 = [height_right; 1; 1];
corner3 = [height_right; width_right; 1];
corner4 = [1; width_right; 1];

corner1p = M * corner1;
corner2p = M * corner2;
corner3p = M * corner3;
corner4p = M * corner4;

maxy = max([corner1(1), corner2(1), corner3(1), corner4(1), corner1p(1), corner2p(1), corner3p(1), corner4p(1)]);
miny = min([corner1(1), corner2(1), corner3(1), corner4(1), corner1p(1), corner2p(1), corner3p(1), corner4p(1)]);
maxx = max([corner1(2), corner2(2), corner3(2), corner4(2), corner1p(2), corner2p(2), corner3p(2), corner4p(2)]);
minx = min([corner1(2), corner2(2), corner3(2), corner4(2), corner1p(2), corner2p(2), corner3p(2), corner4p(2)]);

newHeight = round(maxy - miny);
newWidth = round(maxx - minx);

final_im = zeros(newHeight, newWidth);

for r = 1:size(final_im, 1)
    for c = 1:size(final_im, 2)
        useRightIm = 0;
        useLeftIm = 1;

        canvasloc = [r; c; 1];
        baseloc = canvasloc + [miny - 1; minx - 1; 0];
        % disp(baseloc);
        % disp("\\")

        if baseloc(1) >= 1 && baseloc(2) >= 1 && baseloc(2) <= size(im_right, 2) && baseloc(1) <= size(im_right, 1)
            useRightIm = 1;
        end
        
        otherloc = (inv(M)) * baseloc;
        % disp(otherloc);
        % disp("\\")
        if otherloc(1) < 1 || otherloc(2) < 1 || otherloc(1) > size(im_left, 1) || otherloc(2) > size(im_left, 2)
            useLeftIm = 0;
        end

        if useRightIm == 1 && useLeftIm == 0
            % disp(baseloc);
            % disp("\\")
            final_im(r, c, 1) = im_right(int32(baseloc(1)), int32(baseloc(2)), 1);
            final_im(r, c, 2) = im_right(int32(baseloc(1)), int32(baseloc(2)), 2);
            final_im(r, c, 3) = im_right(int32(baseloc(1)), int32(baseloc(2)), 3);
        end

        if useLeftIm == 1 && useRightIm == 0
            % disp(leftloc);
            % disp("\\")
            final_im(r, c, 1) = im_left(int32(otherloc(1)), int32(otherloc(2)), 1);
            final_im(r, c, 2) = im_left(int32(otherloc(1)), int32(otherloc(2)), 2);
            final_im(r, c, 3) = im_left(int32(otherloc(1)), int32(otherloc(2)), 3);
        end

        if useLeftIm == 1 && useRightIm == 1
            final_im(r, c, 1) = alpha * im_left(int32(otherloc(1)), int32(otherloc(2)), 1) + (1 - alpha) * im_right(int32(baseloc(1)), int32(baseloc(2)), 1);
            final_im(r, c, 2) = alpha * im_left(int32(otherloc(1)), int32(otherloc(2)), 2) + (1 - alpha) * im_right(int32(baseloc(1)), int32(baseloc(2)), 2);
            final_im(r, c, 3) = alpha * im_left(int32(otherloc(1)), int32(otherloc(2)), 3) + (1 - alpha) * im_right(int32(baseloc(1)), int32(baseloc(2)), 3);
        end
    end
end

imshow(final_im);

