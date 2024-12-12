# Introduction:
Fracture detection in X-ray imaging represents a pivotal challenge in medical diagnostics, with significant implications for patient care. Timely and accurate identification of fractures is critical not only for ensuring prompt medical intervention but also for reducing the risk of complications, expediting recovery, and minimizing long-term disability. Despite advancements in medical imaging, reliance on manual interpretation by radiologists often remains a bottleneck, particularly in regions with limited access to trained professionals. Developing an automated and reliable model for fracture detection addresses these challenges by acting as a secondary diagnostic tool, capable of reducing diagnostic errors and alleviating the workload of medical experts.

This study leverages the FracAtlas dataset, a comprehensive collection of X-ray images annotated by a multidisciplinary team of radiologists and orthopedic specialists. The dataset’s meticulous labeling ensures the quality and robustness needed for training state-of-the-art machine learning models. This cool dataset was chosen due to its prelabeled boxes, added to increase our model’s performance. This dataset was selected for its pre-labeled boxes, which are intended to improve the performance of our model. By employing both convolutional neural networks (CNNs and EfficientNet B0) and transformer-based architectures, such as Vision Transformers, the study explores innovative approaches to enhance predictive accuracy. These architectures were chosen for their proven efficacy in computer vision tasks, with CNNs excelling in spatial feature extraction and Vision Transformers demonstrating exceptional performance in capturing global dependencies within images.

The broader implications of developing a highly accurate fracture detection model are profound. In addition to improving diagnostic precision, such models can democratize access to healthcare by supporting clinicians in resource-constrained settings. Furthermore, the integration of deep learning into medical imaging fosters advancements in computer-aided diagnostics, paving the way for a future where AI-driven tools significantly enhance clinical workflows. The outcomes of this study not only underscore the potential of machine learning in transforming medical imaging but also highlight the societal impact of such technologies in improving healthcare delivery worldwide.

# Figures:
The following are a selection of images from the training dataset, along with their corresponding labels. These images represent both fractured and non-fractured bone samples, providing a diverse range of examples for the model to learn from. The labels, which are encoded as '0' for non-fractured and '1' for fractured, are aligned with the images to indicate the correct classification. This visual representation allows us to verify the quality and relevance of the data, ensuring that the model is trained on accurately labeled and high-quality samples. By showcasing these images, we can also highlight the variability in the dataset, which includes differences in bone structures, fracture types, and imaging conditions, further contributing to the robustness of the model training process. 

![image](https://github.com/user-attachments/assets/13b16ef6-9303-465e-ba5c-79ff2ec6d134)

![image](https://github.com/user-attachments/assets/41c191cc-9eb3-4742-9250-b97dcba1ef6c)

![image](https://github.com/user-attachments/assets/f3aa69c2-0fa2-431c-927b-04d9e0ab3553)

Additionally, we have included a few sample outputs of our bounding box visualization function. Bounding boxes are particularly useful because they provide a clear, visual representation of the areas of interest within an image, helping to highlight specific features, which is fractures in bone images in this case. By overlaying the bounding boxes on the images, we can visually verify the accuracy of the object detection process, ensuring that the model is correctly identifying and localizing fractures. This technique is essential for tasks that involve object localization, as it not only improves model interpretability but also aids in evaluating the performance of the model in detecting specific features, which is critical for downstream applications in medical image analysis.

![image](https://github.com/user-attachments/assets/06112fc9-f3bb-424b-af90-f848252287f3)

We decided on a CNN model because convolutional neural networks are particularly well-suited for image classification tasks due to their ability to automatically learn hierarchical features from raw image data. Unlike traditional machine learning models, which rely on manual feature extraction, CNNs can efficiently capture spatial hierarchies and patterns through layers of convolutional filters, making them highly effective for image-based tasks. Their ability to detect local patterns, such as edges, textures, and shapes, in the initial layers and more complex patterns in deeper layers, allows CNNs to generalize well to a variety of image recognition challenges. Additionally, CNNs are computationally efficient and scalable, which makes them ideal for handling large and complex datasets, such as medical images, where subtle differences in image features are crucial for accurate classification. Given the nature of our dataset, which includes diverse bone images with varying fracture patterns, a CNN model offers the robustness and flexibility necessary for achieving high accuracy and reliable results. The following is an example of architecture for the CNN:

![image](https://github.com/user-attachments/assets/e83f1b20-ad44-4d65-931b-0f58d5d7da16)

We are also considering implementing another state of the art model, Vision Transformer (ViT) due to its promising performance in image classification tasks, particularly in domains like medical imaging. Unlike CNNs, which rely on local receptive fields to capture spatial patterns, Vision Transformers excel at capturing long-range dependencies across the entire image thanks to their self-attention mechanisms. These transformers divide images into smaller, fixed-size patches, treating each patch as a sequence input to a transformer model, effectively flattening the image into a sequence of tokens. This unique approach allows the model to learn relationships not only within localized regions of the image but also across the entire spatial context, making it particularly useful for detecting subtle or global patterns that might be overlooked by traditional CNN architectures.
In medical imaging, where complex relationships between features can be critical for accurate diagnosis—such as the interactions between different regions of a bone fracture—ViTs offer an advantage by enabling the model to capture these global dependencies. The ability to model long-range interactions in the data may lead to more robust and precise feature extraction, enhancing the model's performance, especially when dealing with intricate or large-scale images. Given the growing success of Vision Transformers in image classification tasks, especially in fields like medical imaging and remote sensing, we believe that experimenting with this architecture could potentially improve our model's accuracy and provide an alternative approach to the more traditional CNN-based solutions. The vision transformer consists of an encode and position embedding as shown below:

![image](https://github.com/user-attachments/assets/3cae4af8-f6c0-488c-8f23-4c7489ad9c91)

![The-architecture-of-EfficientNetB0-CNN-EfficientNetB0-uses-slightly-larger-mobile](https://github.com/user-attachments/assets/96e3c9d9-4184-48b3-a8d1-e0b97736894f)

# Methods:
## Data Exploration:
### Dataset Description
The FracAtlas dataset is a collection of 4,083 image files, each accompanied by a corresponding text file that contains information about any fractures present within the image. The images represent two categories: fractured bones and non-fractured bones. Specifically, there are 717 images of fractured bones and 3,366 images of non-fractured bones. This class imbalance is important to consider during model training, as it may influence the performance of the machine learning algorithm.

The images in the dataset exhibit a variety of dimensions and color scales. Given this variability, it is crucial to standardize and normalize the images to ensure consistent processing. This variation in size and color scale also highlights the need for careful data preprocessing steps such as resizing, grayscale conversion, and normalization, which were applied to create a more uniform dataset for machine learning tasks.

### Checking for Missing Data
During the data exploration process, we performed a thorough check for missing or inconsistent data. This involved comparing the number of image files to the number of corresponding label files. The two counts matched exactly, confirming that no data was missing from the dataset. Additionally, we verified the consistency of the file names between the image and label files, ensuring that each image had a corresponding annotation, which is critical for supervised learning tasks.

By confirming the completeness and consistency of the dataset, we could proceed with confidence, knowing that the data used for training would be reliable. While downloading these images, however, we found that some of the images were corrupt or truncated which we skipped over while building our train test and validation set and ended up including 4,024 images total.

### Plotting Data
As part of the initial exploratory analysis, we visualized several key aspects of the dataset, including class distribution and image dimensions.

### Class Distribution
We examined the distribution of the two classes—‘Fractured’ and ‘Non_fractured’—and found that the total number of images was consistent across both categories, summing to 4,083 images. The breakdown is as follows:
- 717 images of fractured bones (class 1)
- 3,366 images of non-fractured bones (class 0)

This imbalance is an important consideration when training machine learning models, as it may introduce bias toward the majority class (non-fractured). As a result, techniques like SMOTE (Synthetic Minority Oversampling Technique) were applied to address this issue.

### Image Dimensions
We also investigated the dimensions of the images to assess their consistency. Our analysis revealed that there were 11 unique image sizes within the dataset. The most common image size was 373 pixels in width by 454 pixels in height, which occurred in 2,704 images. This indicates that a significant portion of the dataset shares a common size, which is advantageous for model training since using consistent image dimensions reduces computational complexity.

The presence of varying image sizes across the dataset posed a challenge, which was addressed by resizing all images to a fixed dimension during preprocessing. This ensured that each image had a consistent input shape, which is essential for effective model training.

### Visualization of Images
To gain a better understanding of the dataset and to visually inspect its quality, we randomly selected and displayed five images from the dataset. This visual inspection confirmed that the images were of high quality and relevant to the task at hand, providing useful insights into the nature and variety of the data. Additionally, we plotted the images by class (i.e., ‘Fractured’ or ‘Non_fractured’) to better understand the distribution of the data and verify that the annotations were correct.

### Data Preprocessing for Image Classification
In order to prepare the dataset for the image classification task, a series of preprocessing steps were implemented to ensure that the images were appropriately formatted and standardized, improving the performance of the machine learning model.

### Image Resizing and Grayscale Conversion
To ensure uniformity and consistency across the dataset, all images were resized to a fixed dimension. This step is crucial, as varying image sizes can significantly affect model performance, particularly when training deep learning models. All images were resized to a common resolution (e.g., 224x224 pixels) to ensure that each image had the same input shape, reducing computational complexity and ensuring that the model processes all images in a consistent manner.

Additionally, the images were converted to grayscale to simplify the dataset and reduce computational complexity. The rationale behind converting the images to grayscale is to eliminate the color channels, which are not essential for the current classification task. By doing so, the model can focus on the structural features of the images, such as edges and textures, rather than color information, which can sometimes introduce unnecessary complexity.

### Normalization of Pixel Values
To improve the stability and convergence of the machine learning model during training, all pixel values in the grayscale images were normalized to a range between 0 and 1. This normalization process involved dividing each pixel intensity by 255, the maximum possible pixel value for 8-bit images. This step ensures that the input data is scaled consistently, which is especially important for algorithms that rely on gradient-based optimization methods, such as deep learning networks. Normalizing the data also helps to prevent issues where certain features dominate others due to their larger scale, leading to better overall model performance.

### Feature Standardization
To further enhance the effectiveness of the machine learning algorithm, feature standardization was applied. In this step, the pixel values were standardized such that the mean of the features (i.e., the pixel values) was shifted to 0, and the standard deviation was scaled to 1. This ensures that each feature (in this case, each pixel in the image) has the same scale, which is important for models that assume normally distributed data, such as those using gradient descent. By standardizing the image data, we can improve convergence speed and model accuracy, as the optimization algorithm treats all features equally.

### Label Encoding
To prepare the labels for machine learning, label encoding was performed. This step involved converting the image labels into numerical values, which are required by many machine learning algorithms. Specifically, images were labeled as "Non-fractured" or "Fractured," with "Non-fractured" assigned the label 0, and "Fractured" assigned the label 1. This binary encoding makes it easier for the model to learn and predict the class of each image, particularly when using models that require numerical outputs, such as classification algorithms based on neural networks or decision trees.

### Bounding Box Annotation Visualization
To ensure the correct application of YOLO (You Only Look Once) annotations, a custom function draw_bounding_boxes was implemented. This function visualizes the bounding boxes by overlaying them on the corresponding images. The bounding boxes were drawn based on the YOLO annotation files, which specify the relative coordinates of the bounding boxes in terms of the image's width and height. These relative coordinates were converted to absolute pixel values to accurately position the bounding boxes within the image.
The visualization of bounding boxes serves two purposes: (1) it helps verify the correctness of the annotation process and (2) it enables easy inspection of the dataset to ensure that the bounding boxes are properly aligned with the objects of interest, which in this case may be fractures or other key features.

### Class Imbalance Handling
Given the inherent class imbalance between fractured and non-fractured images with our dataset primarily consisting of non-fractured images, the SMOTE (Synthetic Minority Oversampling Technique) algorithm was applied. SMOTE is a technique used to generate synthetic samples for the minority class by interpolating between existing samples, thereby balancing the number of samples in both the fractured (class 1) and non-fractured (class 0) categories. This step is critical, as imbalanced datasets can lead to biased models that favor the majority class. By applying SMOTE, we ensured that the machine learning model was trained on a more balanced dataset, reducing the risk of overfitting to the majority class and improving the model's ability to generalize to unseen data.

### Final Dataset Preparation
After completing the aforementioned preprocessing steps—resizing, grayscale conversion, normalization, feature standardization, label encoding, bounding box visualization, and class balancing—the dataset was ready for use in model training. The images were consistent in terms of size and format, with pixel values properly scaled and standardized. The labels were numerically encoded, and the dataset was balanced to address class imbalance. These steps were crucial to ensure that the model could learn effectively and make accurate predictions, particularly when dealing with real-world data that may exhibit noise, class imbalance, or varying input formats. By combining these preprocessing techniques, we significantly improved the quality and consistency of the data, making it more suitable for training a robust and effective image classification model.

### Data Variability
As part of our data pre-processing steps, we also want to apply transformations to the images to increase data variability. We can also zoom in or out, change the coordinates of the images, or modify the images through inclusion of noise to further enhance variability for training our model.

## Discussion of Models:
### Model 1 CNN:
We developed and trained two distinct convolutional neural network (CNN) models, iterating through several rounds of hyperparameter tuning to optimize performance. The first baseline model we then optimized consisted of a CNN architecture with three convolutional layers, each employing progressively larger filter sizes. This was followed by batch normalization, ReLU activations, a global average pooling layer, and a dense output layer. During the hyperparameter tuning process, we experimented with variations in the number of filters and the number of convolutional layers to enhance the model's accuracy. The detailed architecture outlined below yielded the best performance in terms of accuracy:

![image](https://github.com/user-attachments/assets/de0435d9-fa8d-4693-bc74-c479c957137a)

We also decided to train a CNN second model, consisting of a sequential neural network, and started with a different set of hyperparameters and model architecture, which we then fine tuned.  The initial layer is a convolutional layer with 64 filters, followed by batch normalization. Then, a second convolutional layer with 128 filters and batch normalization is employed. A global average pooling layer then reduces spatial dimensions. Finally, a dense layer with 512 units feeds into the output layer.

Here is the detailed model architecture for our second model:

![image](https://github.com/user-attachments/assets/3d18f01b-bf39-463c-bb04-f3d7722a45b3)

### Hyperparameter tuning:
Our second model showed promising results, so we decided to implement a hyperparameter tuning process. After preprocessing the data (splitting into training and validation sets, sexising the images to 128x128 pixels to ensure consistency and compatibility with the CNN model), a hyperparameter search is conducted using the RandomSearch tuner. Each configuration trained for 5 epochs wth a batch size of 16. The best hyperparameter combination is determined based on validation accuracy, and early stopping prevents overfitting by halting training if validation loss doesn't improve for 3 consecutive epochs.

Here was our hyperparameter search space:
- conv_1_filters: 16, 32, 48, 64, 80, 96, 112, 128
- conv_1_kernel: 3, 5
- conv_2_filters: 32, 48, 64, 80, 96, 112, 128
- conv_2_kernel: 3, 5
- conv_3_filters: 64, 96, 128
- conv_3_kernel: 3, 5
- dense_units: 64, 128, 192, 256, 320, 384, 448, 512
- optimizer: Adam, SGD
- learning_rate: 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1

Here is the detailed model architecture for our second model after hyperparameter tuning:

![image](https://github.com/user-attachments/assets/788dbd76-7c5c-4da4-b4bb-0ec296b75d1f)

### Model 2 ViT:
The ViT (Vision Transformer) implemented in this project is an advanced model adapted from transformer architecture. Unlike CNNs, the ViT processes images by dividing them into non overlapping patches and treating those patches as a sequence. The model begins by dividing the input image into patches of 16*16, where each patch is like a “token.” The patches layer facilitates this by extracting patches and reshaping them into a sequence of flattened patches. The number of patches, NUM_PATCHES, is derived by dividing the image dimensions by the patch patch size. The patch is then projected into a higher_dimensional space (D_MODEL) using a dense layer in Patch_Encoder. This layer also adds positional embeddings to patches to retain information, as transformers lack sense of locality.

The encoded patches are passed through a series of transformer layers, with NUM_LAYERS = 4. Each layer includes multi-head self-attention (with NUM_HEADS = 4) and a multi-layer perceptron (MLP) block. The self-attention mechanism enables the model to learn global relationships between patches, capturing context across the entire image. Layer normalization and residual connections ensure stable training and enhance feature learning. The MLP block, parameterized by MLP_DIM = 256, further refines the patch representations.

After processing through the transformer layers, a global representation of the image is derived using global average pooling. This pooled representation is fed into a dense classification head with a softmax activation to produce the final class probabilities for NUM_CLASSES, which in this case corresponds to fractured and non-fractured categories. Dropout (DROPOUT_RATE = 0.1) is applied throughout the model to prevent overfitting.

![image](https://github.com/user-attachments/assets/fdda73e4-4a2b-4e29-b294-dd88823f54e1)


### Model 3 EfficientNetB0:

The architecture of EfficientNetB0 is centered around Mobile Inverted Bottleneck Convolution (MBConv) layers, which use depthwise separable convolutions to reduce computational cost without compromising on accuracy. It also incorporates Swish activation, a smoother non-linear activation function that enhances gradient flow and overall performance. The network is structured with an initial convolutional stem, a series of MBConv blocks to extract features at multiple resolutions, and a final classification head that includes global average pooling, dropout, and a dense layer with a sigmoid activation function for binary classification.

EfficientNetB0 leverages pre-trained weights from ImageNet, enabling faster convergence and better generalization on our fracture detection dataset. Compared to the previously implemented CNN and ViT models, EfficientNetB0 provides a complementary approach by combining efficiency, scalability, and strong generalization capabilities.

![image](https://github.com/user-attachments/assets/45c7bc19-b9d5-49d2-8f16-e7a346dd1c16)

# Results:
This will include the results from the methods listed above (C). You will have figures here about your results as well. No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

Due to the iterative nature of the machine learning pipeline, we went back and modified some of the preprocessing and model code from the previous milestones. We then found the following results:

For the baseline CNN model, the training accuracy was approximately 51.09%, with a training loss of 0.8022. The model achieved a validation accuracy of 46.69%, accompanied by a validation loss of 0.8305.
After tuning the hyperparameters, the optimized CNN model achieved a training accuracy of 65.48%, with a training loss of 1.3481. The validation accuracy improved to 63.07%, although the validation loss increased to 1.4335. The test accuracy for the hyperparameter-tuned model was 61.11%, with a corresponding test loss of 1.5177.
The hyperparameters that resulted in the highest validation accuracy are as follows:
- Conv_1 Filters: 48
- Conv_1 Kernel Size: 5
- Conv_2 Filters: 112
- Conv_2 Kernel Size: 5
- Conv_3 Filters: 96
- Conv_3 Kernel Size: 5
- Dense Units: 64
- Optimizer: SGD
- Learning Rate: 0.0028
These hyperparameter values were selected based on their ability to maximize the validation accuracy during the tuning process.

![image](https://github.com/user-attachments/assets/d796acd2-3d33-4398-a6eb-b50befe3beef)

For the Vision Transformer baseline model, the best results were achieved with a patch size of 8, an embedding dimension of 256, and 2 attention heads. The optimal number of layers was 2, with an MLP dimension of 256. A dropout rate of 0.5 and a learning rate of 1e-05 were also found to be the most effective for the model.

Vision Transformer after hyperparameter tuning:

![image](https://github.com/user-attachments/assets/0904eab0-39fb-478b-b996-0eda377cd74f)
```
Trial 5 Complete [00h 06m 08s]
val_accuracy: 0.793398529291153

Best val_accuracy So Far: 0.810513436794281
Total elapsed time: 00h 24m 06s
Best Patch Size: 8
Best Embedding Dimension: 256
Best Number of Heads: 2
Best Number of Layers: 2
Best MLP Dim: 256
Best Dropout Rate: 0.5
Best Learning Rate: 1e-05
```

For the EfficientNet B0 model, the test loss was 0.6928, and the test accuracy was approximately 59.72%. These results highlight the model's performance on the test set, providing an indication of its generalization ability after training. We also computed model metrics including the precision, recall, and a confusion matrix:

![image](https://github.com/user-attachments/assets/5153595a-0f33-4dbd-b975-c3d5870058f5)

![image](https://github.com/user-attachments/assets/d0e9868a-45a3-4ebf-af02-e5bd55655698)

# Discussion:
### CNN baseline model:

The performance of the 1st model on the training set shows a loss of approximate 0.6756 and an accuracy of about 75.22%. The performance of the model on the validation set shows a loss of approximately 0.5254 and an accuracy of about 82.28%. The performance of the model on the testing set shows a loss of approximately 0.6286 and an accuracy of about 78.32%. This suggests that the model might be underfitting as it is not learning the training data well enough, likely because of the imbalance in the two label classes. The dataset primarily consists of non fractured images (around 80%) so we need to ensure the training, test, and validation sets have a more equal split between the classes.

We can improve the model by enhancing its performance on both the training and test sets through tuning hyperparameters. Adding regularization techniques will help prevent overfitting if needed. Modifying the architecture to add more layers can also allow for more complex features to be extracted. Smaller batch sizes can also allow for more complex patterns to be learned, even if it may increase training time. We believe Our model is underfitting on the fitting graph. This indicates that the model is too simple at capturing the underlying patterns in the data, as it only has 3 convolutional layers. Another contributing factor is the imbalance in the dataset, where non-fractured images dominate (around 80%), leading to biased learning that overlooks the minority class.

### VIT model:
To help prevent class imbalances, we tried random oversampling of the minority output label (which is fractured images) in this case. The training accuracy is around 98.88 and the training loss is 0.0353. On the other hand, the validation accuracy is 61.81% while the validation loss is 1.7631. Because there appears to be a significant difference between the training loss and accuracy and the validation loss and accuracy, our model may not be able to generalize well to new unseen data even though there is a very high training accuracy.
We may need to implement techniques such as early stopping (stopping training when there is no significant improvement in validation accuracy after a specific number of epochs) and cross-validation. The model may also be too complex which could also be contributing to the overfitting. Continuing to finetune the model hyperparameters with tools such as GridSearch may also help us find a model architecture that is able to learn the training data well while also being able to generalize to new data.Because training accuracy is in the high 90s while the validation accuracy low 60s after training for 20 epochs, the model seems to be overfitting. The large difference between the training and validation accuracy implies that the model may be overfitting to the specific patterns of the training data and therefore is not able to generalize well. We may need to employ some regularization techniques and continue fine tuning the hyperparameters in order to prevent overfitting.

Comparing the vision transformer model to the first model (the CNN) we trained, the CNN has a higher training and validation accuracy, although that model was underfitting. However, we plan to continue to train both models and improve their accuracy and performance while evaluating other metrics such as precision and recall. We modified the hyperparameters and got the following metrics: 
True Positive: 49
False Positive: 65
True Negative: 263
False Negative: 32
Accuracy: 0.7628361858190709
Recall: 0.6049382716049383
Precision: 0.4298245614035088
F1 Score: 0.5025641025641027
The hyperparameters we used:
patch_size 64
D_model: 512
num_heads 8
Num_layers: 2 
Mlp_dim: 512 
Dropout_rate: 0.3  
learning_rate: 1e-06

Upon further fine tuning of these parameters, 

Best Patch Size: 8
Best Embedding Dimension: 256
Best Number of Heads: 2
Best Number of Layers: 2
Best MLP Dim: 256
Best Dropout Rate: 0.5
Best Learning Rate: 1e-05

We were able to achieve for 1.409806 for test loss and 0.7750611305236816 for Test Accuracy, making it our best performing model yet. 

### EfficientNetB0 model:
The results of the EfficientNetB0 model reveal significant challenges with both training and test performance, suggesting issues of underfitting. During training, the model’s accuracy fluctuated around 50%, and the loss hovered near 0.693, which is indicative of a model that is not learning meaningful features from the data. This is further corroborated by the validation accuracy, which, while initially reaching high values (above 80%), is unstable and potentially misleading due to the model’s inability to separate the two classes effectively, as indicated by the precision and recall metrics.

On the test dataset, the model achieved an overall accuracy of 60% with a macro-averaged F1-score of 0.37. A closer examination of the per-class metrics shows that the model correctly identifies most non-fractured samples (recall of 1.00) but fails entirely on the fractured samples (recall of 0.00). This stark imbalance in performance indicates that the model might be biased towards the majority class (non-fractured), which often occurs in imbalanced datasets.

The EfficientNetB0 model appears to be underfitting the data. This underfitting could be attributed to the freezing of the pre-trained EfficientNetB0 layers, which might prevent the model from adapting its learned features to the fracture detection task, especially given the domain shift from ImageNet data to medical imagery. Additionally, the use of grayscale inputs converted to RGB might lead to information loss, further hampering feature extraction. The persistent high validation accuracy coupled with poor test performance suggests that the data augmentation pipeline might also contribute to over-representing augmented patterns in the training data.

Overall, EfficientNetB0, while offering a more computationally efficient and scalable architecture, showed limitations compared to the Vision Transformer (ViT) in this fracture detection task. Unlike ViT, which achieved smoother learning curves and higher overall validation accuracy, EfficientNetB0 demonstrated significant fluctuations in both training and validation accuracy. Despite these challenges, EfficientNetB0 occasionally achieved competitive peaks in validation accuracy, indicating its potential for generalization. However, it struggled with consistency, as the validation accuracy dropped sharply between epochs. This instability may have stemmed from the model's lighter architecture, which prioritizes efficiency over capacity to handle complex features. In contrast, ViT, with its global attention mechanism, achieved greater stability and better overall performance, particularly in validation accuracy. While EfficientNetB0 is a valuable addition due to its efficiency, its performance highlights the need for additional fine-tuning or adjustments to match the generalization capabilities of ViT.

While the ViT outperformed the CNN after extensive tuning, several challenges remain. The model’s initially high training accuracy underscores the importance of regularization and balanced datasets. The performance gap between fractured and non-fractured classes persists, reflecting the need for more robust techniques, such as focal loss or synthetic data augmentation, to address class imbalance. Additionally, the computational cost of the ViT model is significantly higher than the CNN, raising concerns about scalability and practicality for larger datasets or real-time applications.

Future work could explore hybrid architectures that combine CNNs and transformers to balance efficiency and accuracy. Transfer learning with domain-specific pretraining on medical datasets might further enhance performance. 

# Conclusion:
### Opinions

In conclusion, the Vision Transformer (ViT) model is the best model for our task, as it achieved the highest test accuracy, which is the key metric used for evaluation. This makes it the most reliable choice for future training and model optimization. The ViT model has demonstrated strong performance in generalizing to unseen data, and its test accuracy suggests that with further refinement, it could provide even better results. Given that the test set represents real-world data that the model hasn't encountered during training, a high test accuracy is crucial for ensuring the model's ability to make accurate predictions in practical applications. Therefore, the ViT model is well-suited for continued fine-tuning and further optimization to enhance its performance in future iterations.

### Future Directions
There are several avenues for improvement and exploration that we wish we could have explored to further enhance model performance and broaden its practical applicability. One key direction involves the development of ensemble models that combine the strengths of CNNs and transformers. By integrating these architectures, it is possible to leverage the spatial feature extraction capabilities of CNNs alongside the global dependency modeling of transformers. Such ensemble approaches could reduce the variance and bias inherent to individual models, ultimately boosting predictive accuracy and robustness.

Another promising avenue involves incorporating ConvNext, a state-of-the-art convolutional model inspired by Vision Transformers. ConvNext retains the architectural simplicity of traditional CNNs while incorporating innovations that have propelled transformer architectures to prominence. By exploring ConvNext as a fourth model in future work, it could serve as a strong standalone contender or as a component of an ensemble strategy. Its design, emphasizing efficiency and performance, aligns well with the need for scalable and interpretable solutions in medical imaging.

Looking at the limitations of this study, future work should focus on expanding the dataset by including more samples from different demographics to make the model more widely applicable. In addition, using advanced data augmentation methods and exploring self-supervised learning could help the model perform better with limited labeled data. It would also be beneficial to add features like localization and segmentation to the model, so it can accurately identify fracture locations and provide more detailed diagnostic support.

The potential impact of developing a highly accurate fracture detection model is significant. Beyond enhancing diagnostic accuracy, such models have the potential to increase access to healthcare, particularly in resource-limited settings, by supporting clinicians in their decision-making. Additionally, the use of deep learning in medical imaging drives advancements in computer-aided diagnostics, paving the way for a future where AI-powered tools play a central role in clinical workflows. The findings of this study not only demonstrate the transformative potential of machine learning in medical imaging but also emphasize the broader societal benefits of these technologies in improving healthcare delivery on a global scale.
