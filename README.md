# fracture-detection

## Data Exploratory Steps

### Dataset description
The dataset consists of 717 image files, each with a corresponding text file indicating the location of an observation in the image. There are three subsets in the dataset for training, testing, and validation. These sets consist of 574, 61, and 82 images respectively. The images are not of the same size or color scale, and so they will need to be normalized.

### Checking for missing data
As part of our data exploration process, we assessed the completeness of the data set. We compared the number of image files to the number of label files, which we found to match. We also examined the image and label file names, which we found to be consistent. We noticed that the label files do not all contain the same amount of information, but each file follows a consistent format of five values per line.

## Plotting Data
We analyzed the class distribution and image dimensions within the dataset. We found that the total number of images is consistent across all subsets, amounting to a combined total of [total_images] images. By extracting class IDs from the label files, we identified [number of classes] unique classes present in the dataset. The class IDs are: [sorted class IDs].

We examined the image sizes to assess the consistency of image dimensions. Our analysis revealed [number of unique image sizes] unique image sizes across the dataset. The most common image size is [most_common_size[0]], which occurs in [most_common_size[1]] images. This suggests that while there is some variation in image dimensions, a significant portion of the dataset shares a common size, which may be beneficial for model training.

We randomly selected and displayed five images. This visual inspection confirmed the quality and relevance of the images, providing insights into the variety and nature of the data we are working with.

## Data Pre-processing Steps

### Image Normalization
We want to resize the images to a uniform dimension. We also want to convert our images to grayscale in order to simplify our image dataset and normalize the brightness of the pixels to be in the range between 0 and 1. Additionally, we want to standardize the data so that the mean of the features is at 1 and the standard deviation is at 0. We want to ensure that each image has a consistent format and that each feature has the same scale in order to improve the performance of our machine learning aglorithm.    

### Data Variability
As part of our data pre-processing steps, we also want to apply transformations to the images to increase data variability. We can also zoom in or out, change the coordinates of the images, or modify the images through inclusion of noise to further enhance variability for training our model. 
