**Image Classification & Segmentation of Plant Seedlings**

This project implements various machine learning and deep learning models to classify and segment plant seedling images. It explores preprocessing, architecture design, training strategies, evaluation metrics, and fine-tuning to optimize performance.

------------------
**Data Preprocessing**

The dataset used in this project consists of 409 labeled RGB images representing different categories of plant seedlings. As a first step in the data preprocessing pipeline, random image samples are visualized to assess their quality, clarity, and label correctness. This helps identify any inconsistencies or potential anomalies in the dataset.
Next, all images are reshaped and resized to a standardized input dimension, ensuring compatibility with convolutional neural network (CNN) architectures that require fixed-size inputs.The images are then normalized by scaling pixel values from the original [0, 255] range to a [0, 1] range. This normalization step is essential for faster convergence during training, as it prevents large gradients and stabilizes the optimization process. Finally, the dataset is split into training, validation, and test sets, and the shape of each subset is inspected to confirm the correct data format and distribution.
This structured preprocessing ensures the dataset is clean, balanced, and ready for efficient model training.

----------------------------
**Methodology**

This project employs a combination of classification and segmentation approaches to evaluate model performance on plant seedling imagery.
For classification tasks, several models are implemented and compared, starting with classical machine learning algorithms such as Random Forest, Decision Tree, and Support Vector Machine (SVM). These models serve as baseline predictors and operate on flattened or handcrafted features extracted from the image data.
To improve upon these baselines, we introduce Artificial Neural Networks (ANNs)—simple fully connected feedforward models—that learn non-linear relationships in the data without manual feature engineering.
Building further, the project utilizes Convolutional Neural Networks (CNNs), which are well-suited for image-based tasks due to their ability to automatically learn spatial hierarchies and local patterns through layers of convolutional filters, pooling, and non-linear activations.
In addition to classification, the methodology includes image segmentation models, which aim to identify and localize plant structures within each image. These models are evaluated using metrics like Dice coefficient, precision, and recall, which are critical for assessing performance at the pixel level.
Across all modeling approaches, the dataset is consistently divided into training, validation, and test sets to ensure fair performance comparison and generalization assessment.

------------------------
**Model Architecture**

This project implements two main model architectures tailored for plant seedling image analysis: a U-Net variant using MobileNetV2 as the encoder for segmentation, and a VGG-style deep convolutional network for classification and feature learning.
The segmentation model adopts a U-Net structure where the encoder is based on the pretrained MobileNetV2, initialized with ImageNet weights. Intermediate feature maps from key MobileNet blocks (e.g., block_1_expand_relu, block_3_expand_relu, block_6_expand_relu) are extracted as skip connections. These are progressively concatenated with upsampled feature maps in the decoder. Each decoder block includes Conv2D layers followed by Batch Normalization and ReLU activation, which refine and upsample the spatial representation. The final output layer uses a sigmoid activation function to produce pixel-wise binary segmentation masks.
In parallel, a deep CNN following the VGG-Face architecture is also used. This model stacks multiple blocks of ZeroPadding2D → Conv2D → ReLU → MaxPooling layers, gradually increasing the depth from 64 to 512 filters. The convolutional layers capture increasingly abstract features at deeper levels. After the convolutional base, fully connected Conv2D layers (with kernel sizes 7x7 and 1x1) are applied, followed by Dropout for regularization and a final softmax activation for classification.
These architectures are designed to handle both high-resolution pixel-level segmentation and robust classification, leveraging transfer learning and deep feature hierarchies for optimal performance.

--------------------------
**Model Training**

The training process in this project is designed to optimize both segmentation and classification performance using deep convolutional architectures. For segmentation tasks, the model is trained using binary cross-entropy or Dice loss, which are well-suited for handling imbalanced foreground-background distributions in mask prediction. For classification, categorical cross-entropy is used to differentiate between multiple plant species.
The model training pipeline begins with data normalization, ensuring all pixel values are scaled to the [0, 1] range to stabilize the learning process. Inputs are reshaped to match the expected dimensions of the model ((height, width, 3)), and data is split into training, validation, and test sets. During training, the model uses the Adam optimizer, which adapts the learning rate dynamically and accelerates convergence.
To avoid overfitting, Dropout layers are incorporated into the architecture, and Batch Normalization is applied after convolutions to standardize activations and reduce internal covariate shift. Training is further stabilized using early stopping, which halts training when validation loss no longer improves, and ReduceLROnPlateau, which lowers the learning rate when progress stagnates.
Hyperparameters such as batch size, number of epochs, and initial learning rate are carefully selected and tuned to balance training speed and model generalization. The models are trained for multiple epochs while monitoring training and validation loss, and the best-performing weights are preserved for evaluation on the test set.

--------------------------------
**Hyperparameter Tuning**

To maximize model performance, extensive hyperparameter tuning was conducted across both classical machine learning models and deep learning architectures.
For classical models such as Random Forest and Support Vector Machine (SVM), tuning was performed using GridSearchCV, a method that exhaustively evaluates model performance across a defined parameter grid using cross-validation. In the case of Random Forest, key parameters such as the number of estimators (n_estimators), maximum tree depth (max_depth), and criterion (gini or entropy) were varied to find the optimal configuration. For SVMs, the kernel type (linear, rbf), regularization strength (C), and kernel coefficient (gamma) were tuned. These efforts led to noticeable performance gains—for example, the accuracy of the Random Forest model improved to approximately 42% after tuning.
For the deep learning models, hyperparameters such as learning rate, batch size, and number of epochs were manually adjusted through empirical testing and informed by training curves. The learning rate had a significant impact on training stability, with smaller values improving convergence and reducing overfitting. Dropout rates were also tuned to find a balance between underfitting and overfitting, while batch sizes were adjusted to make effective use of GPU memory without sacrificing generalization.
In combination with early stopping and learning rate schedulers like ReduceLROnPlateau, these tuning strategies contributed to a lower validation loss and improved model robustness across unseen data.

---------------
**Model Evaluation**
The evaluation of the segmentation model was conducted using standard pixel-wise performance metrics, specifically the Dice coefficient, Precision, Recall, and Validation Loss. These metrics offer insights into both the overlap between predicted and ground truth masks and the model's ability to accurately detect relevant regions.




