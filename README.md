# fracture-detection


## Data Exploratory Steps

### Dataset description
The dataset consists of 717 image files, each with a corresponding text file indicating the location of any fracture observations in the image. There are three subsets in the dataset for training, testing, and validation. These sets consist of 574, 61, and 82 images respectively. The images are not of the same size or color scale, and so they will need to be normalized.

### Checking for missing data
As part of our data exploration process, we assessed the completeness of the data set. We compared the number of image files to the number of label files, which we found to match. We also examined the image and label file names, which we found to be consistent. We noticed that the label files do not all contain the same amount of information, but each file follows a consistent format of five values per line. Each line in the label file starts with a "0" or "1", the only valid expected outputs our model should generate.

## Plotting Data
We analyzed the class distribution and image dimensions within the dataset. We found that the total number of images is consistent across all subsets, amounting to a combined total of [total_images] images. By extracting class IDs from the label files, we identified [number of classes] unique classes present in the dataset. The class IDs are: [sorted class IDs].

We examined the image sizes to assess the consistency of image dimensions. Our analysis revealed [number of unique image sizes] unique image sizes across the dataset. The most common image size is [most_common_size[0]], which occurs in [most_common_size[1]] images. This suggests that while there is some variation in image dimensions, a significant portion of the dataset shares a common size, which may be beneficial for model training.

We randomly selected and displayed five images. This visual inspection confirmed the quality and relevance of the images, providing insights into the variety and nature of the data we are working with. We then plotted the images by class ("0" or "1").

## Data Pre-processing Steps

### Image Normalization
We want to resize the images to a uniform dimension. We also want to convert our images to grayscale in order to simplify our image dataset and normalize the brightness of the pixels to be in the range between 0 and 1. Additionally, we want to standardize the data so that the mean of the features is at 1 and the standard deviation is at 0. We want to ensure that each image has a consistent format and that each feature has the same scale in order to improve the performance of our machine learning algorithm.    

### Data Variability
As part of our data pre-processing steps, we also want to apply transformations to the images to increase data variability. We can also zoom in or out, change the coordinates of the images, or modify the images through inclusion of noise to further enhance variability for training our model. 

Link to jupyter notebook: [View the Notebook](eda_milestone2.ipynb)

## Milestone 3
### Where does your model fit in the fitting graph?
Our model is within the ideal range of model complexity as depicted in the fitting graph. This indicates that the model has achieved a balance between underfitting and overfitting, successfully capturing the underlying patterns in the data without being overly sensitive to noise. At this stage, the training error is sufficiently low, reflecting that the model has effectively learned from the dataset, while the test error is also minimized, demonstrating strong generalization to unseen data. The balance in the bias-variance trade-off ensures that the model neither oversimplifies the problem nor becomes excessively complex. The model performs well in precision and accuracy. Being in the ideal complexity range indicates that the model is making reliable predictions. To maintain this performance, we can focus on careful regularization, monitor training and validation performance, fine-tune hyperparameters, and, if feasible, incorporate more diverse training data to further enhance generalization.  

### What are the next models you are thinking of and why?
The next models we're considering are ones that effectively handle image classification and object detection tasks. We plan to implement Convolutional Neural Networks (CNNs) because they are particularly adept at processing image data. CNNs can capture spatial hierarchies and patterns within images through convolutional layers, making them suitable for detecting features indicative of fractures. Starting with a basic CNN architecture will allow us to establish a performance baseline and understand the fundamental capabilities of our dataset.
Building upon that, we are thinking of leveraging transfer learning with pre-trained models such as VGG16, ResNet50, or EfficientNet. These models have been pre-trained on large datasets like ImageNet and have learned rich feature representations that can be fine-tuned for our specific classification task. Using transfer learning can significantly reduce training time and computational resources while potentially improving model performance. By fine-tuning these models on our dataset, we can adapt the learned features to detect fractures more effectively.
Considering that we have bounding box annotations, object detection models like Faster R-CNN, YOLOv5, or SSD are also great. These models not only classify images but also localize objects within them, which is crucial for medical diagnostics where the exact location of a fracture is important. Implementing an object detection model will fully utilize the available annotation data, providing both classification and localization, and offering a more comprehensive analysis of each image.

### New Work/Updates since Milestone 2

Since the last milestone, the project has made significant strides in data preprocessing, dataset preparation, annotation, and building a Convolutional Neural Network (CNN) for image classification. Adjustments were also made to accommodate a new and expanded dataset.

1. **Expanded Dataset**: The original dataset contained only fractured images, which restricted the classification scope. We located an expanded version of the dataset on Kaggle that included both fractured and non-fractured images, enabling a binary classification task.
   
2. **Preprocesssing**: With the new dataset, some preprocessing steps were revisited:
   - *Grayscale Conversion*: All images were resized to 224x224 and converted to grayscale for simplicity and consistency.
   - *Normalization*: Pixel values were normalized to the range [0, 1] for numerical stability.
   - *Standardization*: Images were standardized to have a mean of 0 and a standard deviation of 1.
   - *Label Encoding*: Labels were updated, with Non-fractured images assigned the label 0 and Fractured images assigned the label 1.

 Example ground truth and predictions for train set:
 
 <img width="611" alt="Screenshot 2024-11-17 at 8 17 35 PM" src="https://github.com/user-attachments/assets/ad59a1ab-b071-4786-a663-d899c7e09737">

Example ground truth and predictions for validation set:
<img width="635" alt="Screenshot 2024-11-17 at 8 17 47 PM" src="https://github.com/user-attachments/assets/ca73cb3b-745b-4dea-9afd-e85c75d5177a">

Example ground truth and predictions for test set:

<img width="646" alt="Screenshot 2024-11-17 at 8 17 59 PM" src="https://github.com/user-attachments/assets/e411786f-83a5-4f9e-85b2-7d2be59debcf">

3. **Bounding Box Overlay Function**
   - `draw_bounding_boxes` was implemented to visualize YOLO annotations by overlaying bounding boxes on images using OpenCV
   - The function parsed YOLO annotation files, converted relative bounding box coordinates to absolute pixel values, and drew the boxes on the corresponding images.

4. **Dataset Balancing**: To address class imbalance, the SMOTE (Synthetic Minority Oversampling Technique) algorithm was applied to the training data, ensuring both fractured and non-fractured classes were adequately represented.

5. **Model Development**:

### What is the conclusion of your 1st model? What can be done to possibly improve it?

Our model is a convolutional neural network (CNN) with three convolutional layers featuring increasing filter sizes, followed by batch normalization, ReLU activations, a global average pooling layer, and a dense output layer. The model has a total of 24,394 parameters, of which 24,170 are trainable and 224 are non-trainable.

The performance of the model on the training set shows a loss of approximate 0.6756 and an accuracy of about 75.22%. The performance of the model on the validation set shows a loss of approximately 0.5254 and an accuracy of about 82.28%. The performance of the model on the testing set shows a loss of approximately 0.6286 and an accuracy of about 78.32%. This suggests that the model generalizes well but experiences some overfitting. 

We can improve the model by enhancing its performance on both the training and test sets. Adding regularization techniques will help prevent overfitting. Modifying the architecture to add more layers can also allow for more complex features to be extracted. Smaller batch sizes can also allow for more complex patterns to be learned, even if it may increase training time.

Link to jupyter notebook: [View the Notebook](CNN_notebook.ipynb)
