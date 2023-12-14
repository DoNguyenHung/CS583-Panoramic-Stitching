# CS 583 Final Project: Panoramic Stitching

Hung Do: <hd386@drexel.edu> <br>
Ramona Rubalcava: <rlr92@drexel.edu> <br>

This work is split evenly between Hung and Ramona.

---
## Code Usage:

These MATLAB scripts were written for the purpose of completing the final project for CS 583

#### - Data:

The data used for this assignment were the two images, bikepath_left.jpg and bikepath_right.jpg. Both were personally taken and are not from a website. However, for a reduced processing time the images bikepath_left_resized.jpg and bikepath_right_resized.jpg were used and are the scaled down versions of the original images. 

#### - Folder structure:

The file structure is as follows:

> Final_Project_submission <br>
> ├── intermediate_scripts - folder holding all the intermediate scripts that satisfy parts 1 -5 <br>
> ├── previously_generated_images - folder holding all previously generated images from running the scripts <br>
> ├── bikepath_left.jpg - left image used in stitching <br>
> ├── bikepath_left_resized.jpg- resized image of original image <br>
> ├── bikepath_right.jpg - right image used in stitching <br>
> ├── bikepath_right_resized.jpg - resized image of original image <br>
> └── keypoint_transform.m - matlab script with all parts combined<br>

#### - Running the code:

To recreate the results of this submission, open the script keypoint_transform.m within matlab. Confirm that the images bikepath_left_resized.jpg and bikepath_right_resized.jpg are within the same directory as the script, then run the script. The output of the script should be the final stitched image within a figure titled 'Stitched Image with Transformation Matrix'. 

Within the code there are several figures showing intermediate images during the stitching process. These were commented out in efforts to reduce processing time. To see these intermediate figures, uncomment those lines of code. Also, if overwriting or generating new images is necessary, uncomment the imwrite commands. 

---
