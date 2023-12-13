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
windowSize = int32(size(gray_left, 2) / 80);

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
        tmp_iml = tmp_iml(1:2:end,1:2:end);
        for col=1:size(scales,2) 
            sigma = scales(oct, col); 
            kern = ceil(3*sigma);
            if rem(kern,2)==0
                kern = floor(3*sigma);
            end
            tmp_filter = imgaussfilt(tmp_iml, sigma, FilterSize=kern);
            DoG = tmp_iml - tmp_filter;

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

    % disp("Done octave");
    % disp(oct);
    % disp("\\");

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

        % disp('Done DoG');
        % disp(i);
        % disp("\\");
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

% disp('Done edge pixels')

%% Display all extrema points
gray_left_maximas = gray_left(:, :);
for i = 1:size(maximas, 1)
    gray_left_maximas = insertShape(gray_left_maximas, "circle", [maximas(i, 2),maximas(i, 1), 5], ShapeColor=['red'], Opacity=1);
end

subplot(1, 2, 1), imshow(gray_left_maximas), title('All Extremas');
% disp('Done all extremas');

%% Display pruned extrema points
gray_left_pruned = gray_left(:, :);
is_edge = ismember(maximas, edge_pixels, 'rows');
threshold = mean(stdev_s) + 1.2*std(stdev_s);

count_edge = 0;
count_border = 0;
count_std = 0;
extremas_left = 0;

for i = 1:size(maximas, 1)
    if maximas(i, 1) <= int32(windowSize / 5) || maximas(i, 2) <= int32(windowSize / 5) || maximas(i, 1) > size(gray_left_pruned, 1) - int32(windowSize / 5) || maximas(i, 2) > size(gray_left_pruned, 2) - int32(windowSize / 5)
        count_border = count_border + 1;
        continue
    elseif is_edge(i) == 1
        count_edge = count_edge + 1;
        continue
    elseif stdev_s(i) < threshold
        count_std = count_std + 1;
        continue
    end

    extremas_left = extremas_left + 1;
    gray_left_pruned = insertShape(gray_left_pruned, "circle", [maximas(i, 2),maximas(i, 1), 5], ShapeColor=['red'], Opacity=1);
end

subplot(1, 2, 2), imshow(gray_left_pruned), title('Pruned Extremas');
