%% Reading in the image file
clear all;

im_left = double(imread('bikepath_left_resized.jpg'))/255; 

im_right = double(imread('bikepath_right_resized.jpg'))/255; 

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
          
        end 
    end
end

hold off
saveas(f1, 'image_pyramids.jpg')

%% Downsampled right

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
        end 
    end
end