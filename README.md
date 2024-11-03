# fracture-detection

## Checking for missing data
As part of our data exploration process, we assessed the completeness of the data set. We compared the number of image files to the number of label files, which we found to match. We also examined the image and label file names, which we found to be consistent. We noticed that the label files do not all contain the same amount of information, but each file follows a consistent format of five values per line.

## Image Normalization
We also decided to convert our images to grayscale in order to simply our image dataset and normalized the brightness of the pixels to be in the range between 0 and 1. Additionally, we standardized the data so that the mean of the features is at 1 and the standard deviation is at 0. We wanted to ensure that each image had a consistent format and that each feature had the same scale in order to improve the performance of our machine learning aglorithm.    
