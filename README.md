# fracture-detection

## Checking for missing data
As part of our data exploration process, we assessed the completeness of the data set. We compared the number of image files to the number of label files, which we found to match. We also examined the image and label file names, which we found to be consistent. We noticed that the label files do not all contain the same amount of information, but each file follows a consistent format of five values per line.

## Plotting Data
We analyzed the class distribution and image dimensions within the dataset. We found that the total number of images is consistent across all subsets, amounting to a combined total of [total_images] images. By extracting class IDs from the label files, we identified [number of classes] unique classes present in the dataset. The class IDs are: [sorted class IDs].

We examined the image sizes to assess the consistency of image dimensions. Our analysis revealed [number of unique image sizes] unique image sizes across the dataset. The most common image size is [most_common_size[0]], which occurs in [most_common_size[1]] images. This suggests that while there is some variation in image dimensions, a significant portion of the dataset shares a common size, which may be beneficial for model training.

We randomly selected and displayed five images. This visual inspection confirmed the quality and relevance of the images, providing insights into the variety and nature of the data we are working with.


## Image Normalization
We also decided to convert our images to grayscale in order to simply our image dataset and normalized the brightness of the pixels to be in the range between 0 and 1. Additionally, we standardized the data so that the mean of the features is at 1 and the standard deviation is at 0. We wanted to ensure that each image had a consistent format and that each feature had the same scale in order to improve the performance of our machine learning aglorithm.    
