# fracture-detection

## Milestone 2

### Data Exploratory Steps

#### Dataset description
The dataset consists of 717 image files, each with a corresponding text file indicating the location of any fracture observations in the image. There are three subsets in the dataset for training, testing, and validation. These sets consist of 574, 61, and 82 images respectively. The images are not of the same size or color scale, and so they will need to be normalized.

#### Checking for missing data
As part of our data exploration process, we assessed the completeness of the data set. We compared the number of image files to the number of label files, which we found to match. We also examined the image and label file names, which we found to be consistent. We noticed that the label files do not all contain the same amount of information, but each file follows a consistent format of five values per line. Each line in the label file starts with a "0" or "1", the only valid expected outputs our model should generate.

#### Plotting Data
We analyzed the class distribution and image dimensions within the dataset. We found that the total number of images is consistent across all subsets, amounting to a combined total of [total_images] images. By extracting class IDs from the label files, we identified [number of classes] unique classes present in the dataset. The class IDs are: [sorted class IDs].

We examined the image sizes to assess the consistency of image dimensions. Our analysis revealed [number of unique image sizes] unique image sizes across the dataset. The most common image size is [most_common_size[0]], which occurs in [most_common_size[1]] images. This suggests that while there is some variation in image dimensions, a significant portion of the dataset shares a common size, which may be beneficial for model training.

We randomly selected and displayed five images. This visual inspection confirmed the quality and relevance of the images, providing insights into the variety and nature of the data we are working with. We then plotted the images by class ("0" or "1").

### Data Pre-processing Steps

#### Image Normalization
We want to resize the images to a uniform dimension. We also want to convert our images to grayscale in order to simplify our image dataset and normalize the brightness of the pixels to be in the range between 0 and 1. Additionally, we want to standardize the data so that the mean of the features is at 1 and the standard deviation is at 0. We want to ensure that each image has a consistent format and that each feature has the same scale in order to improve the performance of our machine learning algorithm.    

#### Data Variability
As part of our data pre-processing steps, we also want to apply transformations to the images to increase data variability. We can also zoom in or out, change the coordinates of the images, or modify the images through inclusion of noise to further enhance variability for training our model. 

Link to Milestone 2 Jupyter notebook: [View the Notebook](eda_milestone2.ipynb)

## Milestone 3
### Where does your model fit in the fitting graph?
Our model is underfitting on the fitting graph. This indicates that the model is too simple at capturing the underlying patterns in the data, as it only has 3 convolutional layers. Another contributing factor is the imbalance in the dataset, where non-fractured images dominate (around 80%), leading to biased learning that overlooks the minority class. 


### What are the next models you are thinking of and why?
The next model we are thinking of implementing is the vision transformer because of its performance in image classification tasks. These transformers are suited for capturing long-range dependencies in image data since they have self-attention mechanisms. Vision transformers divide images into patches and treat each patch as a sequence input to a transformer architecture, which is completely different from CNNS. This essentially allows the model to learn relationships across the whole image, making it extremely useful in medical imaging.  

### New Work/Updates since Milestone 2

Since the last milestone, the project has made significant strides in data preprocessing, dataset preparation, annotation, and building a Convolutional Neural Network (CNN) for image classification. Adjustments were also made to accommodate a new and expanded dataset.

1. **Expanded Dataset**: The original dataset contained only fractured images, which restricted the classification scope. We located an expanded version of the dataset that included both fractured and non-fractured images, enabling a binary classification task.
   
2. **Preprocesssing**: With the new dataset, some preprocessing steps were revisited:
   - *Grayscale Conversion*: All images were resized to 224x224 and converted to grayscale for simplicity and consistency.
   - *Normalization*: Pixel values were normalized to the range [0, 1] for numerical stability.
   - *Standardization*: Images were standardized to have a mean of 0 and a standard deviation of 1.
   - *Label Encoding*: Labels were updated, with Non-fractured images assigned the label 0 and Fractured images assigned the label 1.

3. **Bounding Box Overlay Function**
   - `draw_bounding_boxes` was implemented to visualize YOLO annotations by overlaying bounding boxes on images using OpenCV
   - The function parsed YOLO annotation files, converted relative bounding box coordinates to absolute pixel values, and drew the boxes on the corresponding images.

4. **Dataset Balancing**: To address class imbalance, the SMOTE (Synthetic Minority Oversampling Technique) algorithm was applied to the training data, ensuring both fractured and non-fractured classes were adequately represented.

5. **Model Development**: Our model is a convolutional neural network (CNN) with three convolutional layers featuring increasing filter sizes, followed by batch normalization, ReLU activations, a global average pooling layer, and a dense output layer. The model has a total of 24,394 parameters, of which 24,170 are trainable and 224 are non-trainable. 

Example ground truth and predictions for train set:
 
 <img width="611" alt="Screenshot 2024-11-17 at 8 17 35 PM" src="https://github.com/user-attachments/assets/ad59a1ab-b071-4786-a663-d899c7e09737">

Example ground truth and predictions for validation set:
<img width="635" alt="Screenshot 2024-11-17 at 8 17 47 PM" src="https://github.com/user-attachments/assets/ca73cb3b-745b-4dea-9afd-e85c75d5177a">

Example ground truth and predictions for test set:

<img width="646" alt="Screenshot 2024-11-17 at 8 17 59 PM" src="https://github.com/user-attachments/assets/e411786f-83a5-4f9e-85b2-7d2be59debcf">

### What is the conclusion of your 1st model? What can be done to possibly improve it?

The performance of the 1st model on the training set shows a loss of approximate 0.6756 and an accuracy of about 75.22%. The performance of the model on the validation set shows a loss of approximately 0.5254 and an accuracy of about 82.28%. The performance of the model on the testing set shows a loss of approximately 0.6286 and an accuracy of about 78.32%. This suggests that the model might be underfitting as it is not learning the training data well enough, likely because of the imbalance in the two label classes. The dataset primarily consists of non fractured images (around 80%) so we need to ensure the training, test, and validation sets have a more equal split between the classes. 

We can improve the model by enhancing its performance on both the training and test sets through tuning hyperparameters. Adding regularization techniques will help prevent overfitting if needed. Modifying the architecture to add more layers can also allow for more complex features to be extracted. Smaller batch sizes can also allow for more complex patterns to be learned, even if it may increase training time.

Link to Milestone 3 Jupyter notebook: [View the Notebook](CNN_notebook.ipynb)

## Milestone 4
### Where does your model fit in the fitting graph? 
Because training accuracy is in the high 90s while the validation accuracy low 60s after training for 20 epochs, the model seems to be overfitting. The large difference between the training and validation accuracy implies that the model may be overfitting to the specific patterns of the training data and therefore is not able to generalize well. We may need to employ some regularization techniques and continue finetuning the hyperparameters in order to prevent overfitting. 

### What are the next models you are thinking of and why?
The next model we are thinking of implementing is Support Vector Machines (SVMs) which are particularly effective for complex images and finding the optimal decision boundary between our two output classes. Because the SVM model works by finding a hyperplane that maximizes the margin between output classes, we believe there will be high accuracy and precision and also plan to utilize the kernel trick. Additionally, we plan to continue tuning the hyperparameters of both models we have implemented so far - the CNN and vision transformer in order to improve their performance. 

### New Work/Updates since Milestone 3
This milestone, we focused on implementing a new model : the vision transformer and finetuning it to our fracture detection dataset. This model has a total of 1,379,458 parameters, all of which are trainable. Additionally, we have added dropout layers, mutli-head attention, layer normalization and global average pooling to the model. In terms of hyperparameter tuning, we tried various ranges for the different parameters including dropout rate, learning rate, and number of layers. 
|Best Value So Far|Hyperparameter
64                |patch_size
512               |d_model
8                 |num_heads
2                 |num_layers
512               |mlp_dim
0.3               |dropout_rate
1e-06             |learning_rate

### What is the conclusion of your 2nd model? What can be done to possibly improve it?
To help prevent class imbalances, we tried random oversampling of the minority output label (which is fractured images) in this case. The training accuracy is around 98.88 and the training loss is 0.0353. On the other hand, the validation accuracy is 61.81% while the validation loss is 1.7631. Because there appears to be a significant difference between the training loss and accuracy and the validation loss and accuracy, our model may not be able to generalize well to new unseen data even though there is a very high training accuracy. 

We may need to implement techniques such as early stopping (stopping training when there is no significant improvement in validation accuracy after a specific number of epochs) and cross-validation. The model may also be too complex which could also be contributing to the overfitting.  Continuing to finetune the model hyperparameters with tools such as GridSearch may also help us find a model architecture that is able to learn the training data well while also being able to generalize to new data. 

Comparing the vision transformer model to the first model (the CNN) we trained, the CNN has a higher training and validation accuracy, although that model was underfitting. However, we plan to continue to train both models and improve their accuracy and performance while evaluating at other metrics such as precision and recall.

Link to Milestone 3 Jupyter notebook: [View the Notebook](VIT_notebook.ipynb)

