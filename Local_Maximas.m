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

%% Calculate Scales
init_sig = 1.6; 
scales = zeros(4,5);

%auto calculate scales
for row=1:size(scales,1)
    for col=1:size(scales,2)
        scales(row, col) = (2^(row-1))*(sqrt(2)^(col-1))*init_sig;
    end
end

%% Downsampled left
% Find maximas for all octaves and superimpose on the 1st octave (3 x 3)
tmp_iml = gray_left(:,:);
tmp_filter = gray_left(:,:);
windowSize = int32(size(gray_left, 2) / 50);

maximas = [];
stdev_s = [];

hold on
for oct=1:4
    if oct==1
        for col=1:size(scales,2)
            sigma = scales(oct, col); 
            kern = ceil(3*sigma);
            if rem(kern,2)==0
                kern = floor(3*sigma);
            end
            tmp_filter = imgaussfilt(tmp_iml, sigma, FilterSize=kern);
            
            DoG = tmp_iml - tmp_filter;
        end
    else
        tmp_iml = tmp_iml(1:2:end,1:2:end);
        for col=1:size(scales,2) 
            sigma = scales(oct, col); 
            kern = ceil(3*sigma);
            if rem(kern,2)==0
                kern = floor(3*sigma);
            end
            tmp_filter = imgaussfilt(tmp_iml, sigma, FilterSize=kern);
            DoG = tmp_iml - tmp_filter;
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

%% Find edge pixels
edge_left = edge(gray_left);
edge_pixels = [];

for r = 1:size(edge_left, 1)
    for c = 1:size(edge_left, 2)
        if edge_left(r, c) == 1
            edge_pixels = cat(1, edge_pixels, [r, c]);
        end
    end
end

%% Display all extrema points
gray_left_maximas = gray_left(:, :);
for i = 1:size(maximas, 1)
    gray_left_maximas = insertShape(gray_left_maximas, "circle", [maximas(i, 2),maximas(i, 1), 5], ShapeColor=['red'], Opacity=1);
end

subplot(1, 2, 1), imshow(gray_left_maximas), title('All Extremas');

%% Display pruned extrema points
gray_left_pruned = gray_left(:, :);
is_edge = ismember(maximas, edge_pixels, 'rows');
% Need to play around with this more.
threshold = mean(stdev_s) + std(stdev_s);

count_edge = 0;
count_border = 0;
count_std = 0;

for i = 1:size(maximas, 1)
    if maximas(i, 1) <= int32(windowSize / 2) || maximas(i, 2) <= int32(windowSize / 2) || maximas(i, 1) > size(gray_left_pruned, 1) - int32(windowSize / 2) || maximas(i, 2) > size(gray_left_pruned, 2) - int32(windowSize / 2)
        count_border = count_border + 1;
        continue
    elseif is_edge(i) == 1
        count_edge = count_edge + 1;
        continue
    elseif stdev_s(i) < threshold
        count_std = count_std + 1;
        continue
    end

    gray_left_pruned = insertShape(gray_left_pruned, "circle", [maximas(i, 2),maximas(i, 1), 5], ShapeColor=['red'], Opacity=1);
end

subplot(1, 2, 2), imshow(gray_left_pruned), title('Pruned Extremas');
