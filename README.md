# fracture-detection

## Checking for missing data
As part of our data exploration process, we assessed the completeness of the data set. We compared the number of image files to the number of label files, which we found to match. We also examined the image and label file names, which we found to be consistent. We noticed that the label files do not all contain the same amount of information, but each file follows a consistent format of five values per line.

## Dataset description
The dataset consists of 717 image files, each with a corresponding text file indicating the location of an observation in the image. There are three subsets in the dataset for training, testing, and validation. These sets consist of 574, 61, and 82 images respectively. The images are not of the same size or color scale, and so they will need to be normalized.