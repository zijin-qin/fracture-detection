# fracture-detection

## Milestone 3
### Where does your model fit in the fitting graph?
Our model is within the ideal range of model complexity as depicted in the fitting graph. This indicates that the model has achieved a balance between underfitting and overfitting, successfully capturing the underlying patterns in the data without being overly sensitive to noise. At this stage, the training error is sufficiently low, reflecting that the model has effectively learned from the dataset, while the test error is also minimized, demonstrating strong generalization to unseen data. The balance in the bias-variance trade-off ensures that the model neither oversimplifies the problem nor becomes excessively complex. The model performs well in precision and accuracy. Being in the ideal complexity range indicates that the model is making reliable predictions. To maintain this performance, we can focus on careful regularization, monitor training and validation performance, fine-tune hyperparameters, and, if feasible, incorporate more diverse training data to further enhance generalization.  

### What are the next models you are thinking of and why?
The next models we're considering are ones that effectively handle image classification and object detection tasks. We plan to implement Convolutional Neural Networks (CNNs) because they are particularly adept at processing image data. CNNs can capture spatial hierarchies and patterns within images through convolutional layers, making them suitable for detecting features indicative of fractures. Starting with a basic CNN architecture will allow us to establish a performance baseline and understand the fundamental capabilities of our dataset.
Building upon that, we are thinking of leveraging transfer learning with pre-trained models such as VGG16, ResNet50, or EfficientNet. These models have been pre-trained on large datasets like ImageNet and have learned rich feature representations that can be fine-tuned for our specific classification task. Using transfer learning can significantly reduce training time and computational resources while potentially improving model performance. By fine-tuning these models on our dataset, we can adapt the learned features to detect fractures more effectively.
Considering that we have bounding box annotations, object detection models like Faster R-CNN, YOLOv5, or SSD are alsogreat. These models not only classify images but also localize objects within them, which is crucial for medical diagnostics where the exact location of a fracture is important. Implementing an object detection model will fully utilize the available annotation data, providing both classification and localization, and offering a more comprehensive analysis of each image.

### New Work/Updates since Milestone 2

The `draw_bounding_boxes` function was added to the preprocessing stage which visualizes YOLO annotations by overlaying bounding boxes on images. It reads YOLO-formatted annotation files, converts relative coordinates to pixel values, and draws the corresponding boxes on the image using OpenCV. 

### What is the conclusion of your 1st model? What can be done to possibly improve it?

Our model is a convolutional neural network (CNN) with three convolutional layers featuring increasing filter sizes, followed by batch normalization, ReLU activations, a global average pooling layer, and a dense output layer. The model has a total of 24,394 parameters, of which 24,170 are trainable and 224 are non-trainable.

The performance of the model on the training set shows a loss of approximate 0.6756 and an accuracy of about 75.22%. The performance of the model on the validation set shows a loss of approximately 0.5254 and an accuracy of about 82.28%. The performance of the model on the testing set shows a loss of approximately 0.6286 and an accuracy of about 78.32%. This suggests that the model generalizes well but experiences some overfitting. 

We can improve the model by enhancing its performance on both the training and test sets. Adding regularization techniques will help prevent overfitting. 
