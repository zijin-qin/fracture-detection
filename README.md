# Introduction:
Fracture detection in X-ray imaging represents a pivotal challenge in medical diagnostics, with significant implications for patient care. Timely and accurate identification of fractures is critical not only for ensuring prompt medical intervention but also for reducing the risk of complications, expediting recovery, and minimizing long-term disability. Despite advancements in medical imaging, reliance on manual interpretation by radiologists often remains a bottleneck, particularly in regions with limited access to trained professionals. Developing an automated and reliable model for fracture detection addresses these challenges by acting as a secondary diagnostic tool, capable of reducing diagnostic errors and alleviating the workload of medical experts.

This study leverages the FracAtlas dataset, a comprehensive collection of X-ray images annotated by a multidisciplinary team of radiologists and orthopedic specialists. The dataset’s meticulous labeling ensures the quality and robustness needed for training state-of-the-art machine learning models. This cool dataset was chosen due to its prelabeled boxes, added to increase our model’s performance. This dataset was selected for its pre-labeled boxes, which are intended to improve the performance of our model. By employing both convolutional neural networks (CNNs and EfficientNet B0) and transformer-based architectures, such as Vision Transformers, the study explores innovative approaches to enhance predictive accuracy. These architectures were chosen for their proven efficacy in computer vision tasks, with CNNs excelling in spatial feature extraction and Vision Transformers demonstrating exceptional performance in capturing global dependencies within images.

The broader implications of developing a highly accurate fracture detection model are profound. In addition to improving diagnostic precision, such models can democratize access to healthcare by supporting clinicians in resource-constrained settings. Furthermore, the integration of deep learning into medical imaging fosters advancements in computer-aided diagnostics, paving the way for a future where AI-driven tools significantly enhance clinical workflows. The outcomes of this study not only underscore the potential of machine learning in transforming medical imaging but also highlight the societal impact of such technologies in improving healthcare delivery worldwide.

# Figures:
Your report should include relevant figures of your choosing to help with the narration of your story, including legends (similar to a scientific paper). For reference you search machine learning and your model in google scholar for reference examples.

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
