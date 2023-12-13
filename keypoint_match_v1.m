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

f8 = figure('Name', 'Together');
imshow(together);

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
    windowSize = int32(size(im, 2) / 80);
    
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
           end 
        end
    
        % Find local maximas
        for r = 1:size(DoG, 1)
            for c = 1:size(DoG, 2)
                if mod(r, windowSize) == 1 && mod(c, windowSize) == 1
                    if r + (windowSize - 1) > size(DoG, 1)
                        end_r = size(DoG, 1);
                    else
                        end_r = r + (windowSize - 1);
                    end
    
                    if c + (windowSize - 1) > size(DoG, 2)
                        end_c = size(DoG, 2);
                    else
                        end_c = c + (windowSize - 1);
                    end
    
                    temp = DoG(r:end_r, c:end_c);
    
                    % Calculate std here 
                    stdev = std(temp(:));
                    stdev_s(end + 1) = stdev;
    
                    [val, index] = max(temp, [], "all");
                    [maxima_r, maxima_c] = ind2sub(size(temp), index);
    
                    maximas = cat(1, maximas, [(r + (maxima_r - 1))* 2^(oct - 1), (c + (maxima_c - 1))* 2^(oct - 1)]);
    
                end
            end
        end
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
    threshold = mean(stdev_s) + std(stdev_s);
    
    pruned_maximas = [maximas zeros(size(maximas,1), 1)];

    count_edge = 0;
    count_border = 0;
    count_std = 0;
    
    for i = 1:size(maximas, 1)
        if maximas(i, 1) <= int32(windowSize / 2) || maximas(i, 2) <= int32(windowSize / 2) || maximas(i, 1) > size(pruned_im, 1) - int32(windowSize / 2) || maximas(i, 2) > size(pruned_im, 2) - int32(windowSize / 2)
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
                if temp(i,5) > temp(1,1)
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
    for j=1:size(sorted,1)
        if j==1
            if sorted(j, 3) == sorted(j+1,3)
                if sorted(j,5) > sorted(1,1)
                    fully_filtered = cat(1, fully_filtered, sorted(j,:)); 
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