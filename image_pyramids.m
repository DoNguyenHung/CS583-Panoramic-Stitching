%% Reading in the image file
clear all;

im_left = double(imread('bikepath_left_resized.jpg'))/255; 
%f1 = figure('Name', 'Bike Path Left');
%imshow(im_left)

im_right = double(imread('bikepath_right_resized.jpg'))/255; 
%f2 = figure('Name', 'Bike Path Right');
%imshow(im_right)

%% Grayscale

gray_left = rgb2gray(im_left);
gray_right = rgb2gray(im_right);

%f3 = figure('Name', 'Bike Path Grays');
%subplot(1,2,1), imshow(gray_left);
%subplot(1,2,2), imshow(gray_right);

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

tmp_iml = gray_left(:,:);
tmp_filter = gray_left(:,:);

f1 = figure('Name', 'Bike Path Image Pyramid');

hold on
pos = 1;
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

            subplot(4,5,pos), imshow(tmp_filter), axis on;
            pos = pos+1;
            
            %imwrite(tmp_filter, sprintf('./im_pyramids_bikeleft/GAUSoct%d_scale%d.jpg', oct, col));
            %imwrite(DoG, sprintf('./im_pyramids_bikeleft/DoG_oct%d_scale%d.jpg', oct, col));
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
             
            subplot(4,5,pos), imshow(tmp_filter), axis on;
            pos = pos+1;
          
            %imwrite(tmp_filter, sprintf('./im_pyramids_bikeleft/GAUSoct%d_scale%d.jpg', oct, col));
            %imwrite(DoG, sprintf('./im_pyramids_bikeleft/DoG_oct%d_scale%d.jpg', oct, col));
        end 
    end
end

hold off
saveas(f1, 'image_pyramids.jpg')

% Downsampled right

tmp_imr = gray_right(:,:);
tmp_filt = gray_right(:,:);

for oct=1:4
    if oct==1
        for col=1:size(scales,2)
            sigma = scales(oct, col); 
            kern = ceil(3*sigma);
            if rem(kern,2)==0
                kern = floor(3*sigma);
            end
            tmp_filt = imgaussfilt(tmp_imr, sigma, FilterSize=kern);
            DoG = tmp_imr - tmp_filt;
            %imwrite(tmp_filt, sprintf('./im_pyramids_bikeright/GAUSoct%d_scale%d.jpg', oct, col));
            %imwrite(DoG, sprintf('./im_pyramids_bikeright/DoG_oct%d_scale%d.jpg', oct, col));
        end 
    else
        tmp_imr = tmp_imr(1:2:end,1:2:end);
        for col=1:size(scales,2) 
            sigma = scales(oct, col); 
            kern = ceil(3*sigma);
            if rem(kern,2)==0
                kern = floor(3*sigma);
            end
            tmp_filt = imgaussfilt(tmp_imr, sigma, FilterSize=kern);
            DoG = tmp_imr - tmp_filt;
            %imwrite(tmp_filt, sprintf('./im_pyramids_bikeright/GAUSoct%d_scale%d.jpg', oct, col));
            %imwrite(DoG, sprintf('./im_pyramids_bikeright/DoG_oct%d_scale%d.jpg', oct, col));
        end 
    end
end