%% Reading in the image file
clear all;

im_left = double(imread('bikepath_left_resized.jpg'))/255; 
im_left = imresize(im_left, 0.6); 
% im_left = double(imread('sample_left.jpg'))/255; 

im_right = double(imread('bikepath_right_resized.jpg'))/255; 
im_right = imresize(im_right, 0.6);
% im_right = double(imread('sample_right.jpg'))/255; 

%% Grayscale
gray_left = rgb2gray(im_left);
gray_right = rgb2gray(im_right);

%% Calculate Scales and Maximas

scales = calc_scales(1.6);

[max_left, stdev_l, window_l] = scale_max(gray_left, scales);
[max_right, stdev_r, window_r] = scale_max(gray_right, scales);

edge_left = find_edge(gray_left);
edge_right = find_edge(gray_right);

[pruned_left, pm_left] = pruned_max(gray_left, max_left, stdev_l, window_l, edge_left);
[pruned_right, pm_right]= pruned_max(gray_right, max_right, stdev_r, window_r, edge_right);

%% Display Points
%gray_left_maximas = gray_left(:, :);
%for i = 1:size(max_left, 1)
%    gray_left_maximas = insertShape(gray_left_maximas, "circle", [max_left(i, 2),max_left(i, 1), 5], ShapeColor=['red'], Opacity=1);
%end
%
%f1 = figure('Name', 'Extremas Before and After Pruning: Left');
%subplot(1, 2, 1), imshow(gray_left_maximas), title('All Extremas');
%subplot(1, 2, 2), imshow(pruned_left), title('Pruned Extremas');
%
%gray_right_maximas = gray_right(:, :);
%for i = 1:size(max_right, 1)
%    gray_right_maximas = insertShape(gray_right_maximas, "circle", [max_right(i, 2),max_right(i, 1), 5], ShapeColor=['red'], Opacity=1);
%end
%
%f2 = figure('Name', 'Extremas Before and After Pruning: Right');
%subplot(1, 2, 1), imshow(gray_right_maximas), title('All Extremas');
%subplot(1, 2, 2), imshow(pruned_right), title('Pruned Extremas');

%% Point Correspondences

% this returns the similarities for the left and right images 
[lftsims, rgtsims] = keypoints(im_left, im_right, pm_left, pm_right);

% temporary variable that keeps only the rows that are the same 
% between the two variables above
 same_max = lftsims(ismember(lftsims,rgtsims,'rows'),:);

 corrs = filtered(same_max);

%% Plot Correspondences

left_corr = im_left(:, :, :);
right_corr = im_right(:, :, :);
together = [left_corr right_corr];

for i = 1:size(corrs, 1)
    together = insertShape(together, "line", [corrs(i, 2), corrs(i, 1), size(left_corr,2)+corrs(i, 4),corrs(i, 3)], ShapeColor=['red'], LineWidth=2,Opacity=1);
end

%f3 = figure('Name', 'Together');
%imshow(together);

% uncomment below to visualize the individual matched correspondences
%for i = 1:size(corrs, 1)
 %   left_corr = insertShape(left_corr, "circle", [corrs(i, 2),corrs(i, 1), 5], ShapeColor=['green'], Opacity=1);
 %   right_corr = insertShape(right_corr, "circle", [corrs(i, 4),corrs(i, 3), 5], ShapeColor=['green'], Opacity=1);
%end

%f4 = figure('Name', 'Extremas found in both left and right images');
%subplot(1, 2, 1), imshow(left_corr), title('left correspondences');
%subplot(1, 2, 2), imshow(right_corr), title('right correspondences');

%% RANSAC

% creating the random sample of 4 correspondences 
rand = randperm(size(corrs,1),4);
sample = [corrs(rand(1),:); corrs(rand(2),:); corrs(rand(3),:); corrs(rand(4),:)];

stitched_im = trans_matrix(sample, im_left, im_right);

f5 = figure('Name', 'Stitched Image with Transformation Matrix');
imshow(stitched_im)

%% Functions

function scales = calc_scales(init_sig)

    scales = zeros(4,5);
    
    %auto calculate scales
    for row=1:size(scales,1)
        for col=1:size(scales,2)
            scales(row, col) = (2^(row-1))*(sqrt(2)^(col-1))*init_sig;
        end
    end

end

function [maximas, stdev_s, windowSize] = scale_max(im, scales)
    % Find maximas for all octaves and superimpose on the 1st octave (3 x 3)
    tmp_im = im(:,:);
    tmp_filter = im(:,:);
    windowSize = int32(size(im, 2) / 50);
    
    maximas = [];
    stdev_s = [];
    
    for oct=1:4
        if oct==1
            for col=1:size(scales,2)
                sigma = scales(oct, col); 
                kern = ceil(3*sigma);
                if rem(kern,2)==0
                    kern = floor(3*sigma);
                end
                tmp_filter = imgaussfilt(tmp_im, sigma, FilterSize=kern);
    
                DoG = tmp_im - tmp_filter;
                if col == 1
                    DoG_1 = DoG;
                elseif col == 2
                    DoG_2 = DoG;
                elseif col == 3
                    DoG_3 = DoG;
                elseif col == 4
                    DoG_4 = DoG;
                else
                    DoG_5 = DoG;
                end
            end
        else
            tmp_im = tmp_im(1:2:end,1:2:end);
            for col=1:size(scales,2) 
                sigma = scales(oct, col); 
                kern = ceil(3*sigma);
                if rem(kern,2)==0
                    kern = floor(3*sigma);
                end
                tmp_filter = imgaussfilt(tmp_im, sigma, FilterSize=kern);
                DoG = tmp_im - tmp_filter;

                if col == 1
                    DoG_1 = DoG;
                elseif col == 2
                    DoG_2 = DoG;
                elseif col == 3
                    DoG_3 = DoG;
                elseif col == 4
                    DoG_4 = DoG;
                else
                    DoG_5 = DoG;
                end
           end 
        end
    
        % Find local maximas
        for i = 2:4
            curr_DoG = [];
            if i == 2
                curr_DoG(:, :, 1) = DoG_1;
                curr_DoG(:, :, 2) = DoG_2;
                curr_DoG(:, :, 3) = DoG_3;
            elseif i == 3
                curr_DoG(:, :, 1) = DoG_2;
                curr_DoG(:, :, 2) = DoG_3;
                curr_DoG(:, :, 3) = DoG_4;
            elseif i == 4
                curr_DoG(:, :, 1) = DoG_3;
                curr_DoG(:, :, 2) = DoG_4;
                curr_DoG(:, :, 3) = DoG_5;
            end

            % for r = 2:size(curr_DoG, 1) - 1
            %     for c = 2:size(curr_DoG, 2) - 1
            for r = 4:size(curr_DoG, 1) - 3
                for c = 4:size(curr_DoG, 2) - 3
                    temp = curr_DoG(r-3:r+3, c-3:c+3, :);
                    % temp = curr_DoG(r-1:r+1, c-1:c+1, :);
                
                    % Calculate std here
                    reshaped_temp = reshape(temp, [size(temp, 1) * size(temp, 2) * size(temp, 3), 1]);
                    stdev = std(reshaped_temp(:));
    
                    [maxVal, index] = max(reshaped_temp, [], "all");

                    if maxVal == curr_DoG(r, c, 2)
                        [maxima_r, maxima_c] = ind2sub(size(temp), index);
                        stdev_s(end + 1) = stdev;
        
                        maximas = cat(1, maximas, [(r + (maxima_r - 1)) * 2^(oct - 1), (c + (maxima_c - 1))* 2^(oct - 1)]);
                    end
                end
            end
        end
        % disp('Done DoG');
        % disp(i);
        % disp("\\");
    end
end

function edge_pixels = find_edge(im)
    edge_im = edge(im);
    edge_pixels = [];

    for r = 1:size(edge_im, 1)
        for c = 1:size(edge_im, 2)
            if edge_im(r, c) == 1
                edge_pixels = cat(1, edge_pixels, [r, c]);
            end
        end
    end
end

function [pruned_im, pruned_maximas] = pruned_max(im, maximas, stdev_s, windowSize, edge_pixels)
    pruned_im = im(:, :);
    is_edge = ismember(maximas, edge_pixels, 'rows');
    % Need to play around with this more.
    threshold = mean(stdev_s) + 1.2*std(stdev_s);
    
    pruned_maximas = [maximas zeros(size(maximas,1), 1)];

    count_edge = 0;
    count_border = 0;
    count_std = 0;
    extremas_left = 0;
    
    for i = 1:size(maximas, 1)
        if maximas(i, 1) <= int32(windowSize / 5) || maximas(i, 2) <= int32(windowSize / 5) || maximas(i, 1) > size(pruned_im, 1) - int32(windowSize / 5) || maximas(i, 2) > size(pruned_im, 2) - int32(windowSize / 5)
            count_border = count_border + 1;
            %pruned_maximas(i, 3) = 1; % added this
            continue
        elseif is_edge(i) == 1
            count_edge = count_edge + 1;
            %pruned_maximas(i, 3) = 1; % added this
            continue
        elseif stdev_s(i) < threshold
            count_std = count_std + 1;
            %pruned_maximas(i, 3) = 1; % added this
            continue
        end

        pruned_maximas(i, 3) = 1;
        extremas_left = extremas_left + 1;
        pruned_im = insertShape(pruned_im, "circle", [maximas(i, 2),maximas(i, 1), 5], ShapeColor=['red'], Opacity=1);
    end
end

function [left2right, right2left] = keypoints(imagel, imager, pruned_m1, pruned_m2)
    % these two variables reduces the maximas of the left and right to just
    % those that were kept after the pruning process
    kept_l = find(pruned_m1(:,3)==1);
    kept_r = find(pruned_m2(:,3)==1);

    left2right = []; % variable to store the left to right similarities 
    right2left = []; % variable to store the right to left similarities
    

    for row=1:size(kept_l, 1)
        maxleft = kept_l(row,1); % this retrieves the row index for the 'kept' maxima
        leftx = pruned_m1(maxleft, 2); % this grabs the maxima 
        lefty = pruned_m1(maxleft, 1);
        for ind=1:size(kept_r, 1)
            maxright = kept_r(ind,1); % this retrieves the row index for the 'kept' maxima
            rightx = pruned_m2(maxright, 2);
            righty = pruned_m2(maxright,1);
            % for now this checks that the y values within 30 pixels of each other in
            % efforts to reduce the number of correspondences
            if lefty >= righty-30 & lefty <= righty+30
                % the if statements makes sure there is enough surrounding
                % the pixel to calc the 9x9 region
                if (lefty > 4 & (lefty < size(imagel,1)-4)) & (leftx > 4 & (leftx < size(imagel,2)-4))
                    if (righty > 4 & (righty < size(imager,1)-4)) & (rightx > 4 & (rightx < size(imager,2)-4))
                        %sprintf('left y:%d, x:%d',lefty,leftx)
                        region_lft = imagel(lefty-4:lefty+4, leftx-4:leftx+4,:);
                        flatleft = reshape(region_lft, [1,243]);
                        %sprintf('right y:%d, x:%d',righty,rightx)
                        region_rgt = imager(righty-4:righty+4, rightx-4:rightx+4,:);
                        flatright = reshape(region_rgt, [1,243]);

                        sim = dot(flatleft,flatright)/(norm(flatleft)*norm(flatright));
                        if sim > 0.98 % can play with this value
                            left2right = cat(1, left2right, [lefty,leftx,righty,rightx, sim]);
                        end
                    end
                end
            end
        end
    end

    % does the same as above but right to left 
    for ind=1:size(kept_r, 1)
            maxright = kept_r(ind,1);
            rightx = pruned_m2(maxright, 2);
            righty = pruned_m2(maxright,1);
        for row=1:size(kept_l, 1)
            maxleft = kept_l(row,1);
            leftx = pruned_m1(maxleft, 2);
            lefty = pruned_m1(maxleft, 1);
            if righty >= lefty-30 & righty <=lefty+30
                if (righty > 4 & (righty < size(imager,1)-4)) & (rightx > 4 & (rightx < size(imager,2)-4))
                    if (lefty > 4 & lefty < size(imagel,1)-4) & (leftx > 4 & leftx < size(imagel,2)-4)
                        region_rgt = imager(righty-4:righty+4, rightx-4:rightx+4,:);
                        flatright = reshape(region_rgt, [1,243]);
                        
                        region_lft = imagel(lefty-4:lefty+4, leftx-4:leftx+4,:);
                        flatleft = reshape(region_lft, [1,243]);
                        
                        sim = dot(flatright,flatleft)/(norm(flatright)*norm(flatleft));
                        if sim > 0.98 % can play with this value
                            right2left = cat(1, right2left, [lefty,leftx,righty,rightx, sim]);
                        end
                    end
                end
            end
        end
    end
end

function fully_reduced = filtered(corr_matrix)
    % function that selects the highest correspondence based on
    % the cosine similarity value for instances where the correspondences
    % are 'clustered'

    temp = sortrows(corr_matrix);
    reduced = [];
    fully_reduced = []; 
    for i=1:size(temp, 1)
        if i==1
            if temp(i, 1) == temp(i+1,1)
                %disp(temp(i, 1))
                %disp(temp(i+1,1))
                %disp(temp(1,1))
                if temp(i,5) > temp(i+1,5)
                    reduced = cat(1, reduced, temp(i,:)); 
                end
            else
                reduced = cat(1, reduced, temp(i,:));
            end
        elseif temp(i, 1) == reduced(end,1)
            if temp(i,5) > reduced(end,5)
                reduced(end,:) = temp(i,:);
            end
        else
            reduced = cat(1, reduced, temp(i,:));
        end
    end

    sorted = sortrows(reduced,3);
    %disp(sorted)
    for j=1:size(sorted,1)
        if j==1
            if sorted(j, 3) == sorted(j+1,3)
                if sorted(j,5) > sorted(j+1,5)
                    fully_filtered = cat(1, fully_filtered, sorted(j,:)); 
                else
                    fully_reduced = cat(1, fully_reduced, sorted(j+1,:));
                end
            else
                fully_reduced = cat(1, fully_reduced, sorted(j,:));
            end
        elseif sorted(j, 3) == fully_reduced(end,3)
            if sorted(j,5) > fully_reduced(end,5)
                fully_reduced(end,:) = sorted(j,:);
            end
        else
            fully_reduced = cat(1, fully_reduced, sorted(j,:));
        end
    end
end

function final_im = trans_matrix(cmatrix, im_left, im_right)

%    cmatrix: the matrix of the randomly sampled correspondences
%             formated (left y, left x, right y, right x
    alpha = 0.3;
    width_right = size(im_right, 2);
    height_right = size(im_right, 1);
    
    A = [cmatrix(1, :) 1];
    B = [cmatrix(2, :) 1];
    C = [cmatrix(3, :) 1];
    D = [cmatrix(4, :) 1];
    
    %Ap = [cmatrix(1, :) 1];
    %Bp = [cmatrix(2, :) 1];
    %Cp = [cmatrix(3, :) 1];
    %Dp = [cmatrix(4, :) 1];
    
    Am = zeros(8, 6);
    Am(1, :) = [A(1) A(2) 1 0 0 0];
    Am(2, :) = [0 0 0 A(1) A(2) 1];
    Am(3, :) = [B(1) B(2) 1 0 0 0];
    Am(4, :) = [0 0 0 B(1) B(2) 1];
    Am(5, :) = [C(1) C(2) 1 0 0 0];
    Am(6, :) = [0 0 0 C(1) C(2) 1];
    Am(7, :) = [D(1) D(2) 1 0 0 0];
    Am(8, :) = [0 0 0 D(1) D(2) 1];
    
    b = [A(3); A(4); B(3); B(4); C(3); C(4); D(3); D(4)];
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
    
end