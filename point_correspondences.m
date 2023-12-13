%% Reading in the image file
clear all;

ogleft = double(imread('bikepath_left.jpg'))/255; 

ogright = double(imread('bikepath_right.jpg'))/255; 

%% Resize to smaller scale for processing

left = imresize(ogleft, 0.25); 

right = imresize(ogright, 0.25);

%% Point Correspondence with insert 

% the resized locations left (X,Y)
branchlft= [950 245];
rocklft= [853 829];
signlft= [598 730];
leaflft= [1044 1273];

% the resized locations right (X,Y)
branchrgt= [606 319];
rockrgt= [511 850];
signrgt= [259 750];
leafrgt= [628 1258];

withptslft = insertShape(left, "filled-circle", [branchlft(1),branchlft(2),10], ShapeColor=['red'], Opacity=1);
withptslft = insertShape(withptslft, "filled-circle", [rocklft(1),rocklft(2),10], ShapeColor=['magenta'], Opacity=1);
withptslft = insertShape(withptslft, "filled-circle", [signlft(1),signlft(2),10], ShapeColor=['green'], Opacity=1);
withptslft = insertShape(withptslft, "filled-circle", [leaflft(1),leaflft(2),10], ShapeColor=['cyan'], Opacity=1);

withptsrgt = insertShape(right, "filled-circle", [branchrgt(1),branchrgt(2),10], ShapeColor=['red'], Opacity=1);
withptsrgt = insertShape(withptsrgt, "filled-circle", [rockrgt(1),rockrgt(2),10], ShapeColor=['magenta'], Opacity=1);
withptsrgt = insertShape(withptsrgt, "filled-circle", [signrgt(1),signrgt(2),10], ShapeColor=['green'], Opacity=1);
withptsrgt = insertShape(withptsrgt, "filled-circle", [leafrgt(1),leafrgt(2),10], ShapeColor=['cyan'], Opacity=1);

f8 = figure('Name', 'Together');
together = [withptslft, withptsrgt];
imshow(together);