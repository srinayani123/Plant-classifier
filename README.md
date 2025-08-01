Plant Seedling Classification:

This project explores multiple machine learning and deep learning techniques to classify plant seedlings from image data. Approaches include classical supervised models, ANNs, and CNNs. Performance is evaluated across models to determine the most effective method.

📦 Data Preprocessing
The dataset contains RGB images of plant seedlings belonging to various species. Preprocessing steps include:

Resizing images to a fixed shape suitable for CNN input.

Normalization of pixel values by dividing RGB values by 255.

Reshaping images to fit the input format expected by TensorFlow/Keras: (n_samples, height, width, channels).

Label encoding to convert categorical species labels into numerical values.

📊 EDA Analysis
Exploratory Data Analysis (EDA) includes:

Visual inspection of sample images across classes.

Class distribution visualization to identify any imbalance.

Image resolution checks and channel consistency (ensuring RGB format).

Summary statistics on pixel intensity and color distribution.

⚙️ Methodology
The project compares the performance of:

Classical models: Support Vector Machines (SVM), Random Forest, Decision Tree

Artificial Neural Networks (ANNs)

Convolutional Neural Networks (CNNs)

Train-validation-test splits are used to evaluate generalization. Feature extraction is automatic in deep learning models and manual (e.g., PCA) in classical models.

🧠 Model Architecture
CNNs are used with multiple Conv2D, ReLU, Dropout, and MaxPooling layers.

Automatic hierarchical feature learning improves performance and robustness.

ANNs require flattened inputs and do not exploit spatial structure.

Classical models rely on handcrafted or PCA-derived features.

CNNs achieved the best performance with ~79% accuracy and ~0.8 loss.

🏋️ Training
The model is trained using cross-entropy loss and Adam optimizer.

ReLU is used to mitigate vanishing gradients from sigmoid/tanh activations.

Training involves batch normalization and dropout to stabilize and regularize.

Early stopping can be used to prevent overfitting.

Models are evaluated on training and validation datasets after each epoch.

🔧 Finetuning (Hyperparameter Tuning)
GridSearchCV is used to optimize hyperparameters for classical models.

For Random Forest, hyperparameter tuning improved accuracy to ~42%.

For SVM, hyperparameter tuning with GridSearchCV improved accuracy over default SVM.

Parameters tuned include:

Number of estimators (for RF)

Kernel types, C, gamma (for SVM)

CNN hyperparameters (e.g., learning rate, batch size) can also be tuned manually or via tools like Keras Tuner (not shown).

📈 Evaluation
Evaluation metrics include:

Accuracy on validation/test sets

Confusion Matrix for class-specific performance

Loss curves for training vs validation monitoring

Model comparison across different approaches:

Decision Tree: ~15%

Random Forest: ~43%

ANN: ~60%

CNN: ~79%

📤 Output Analysis
CNN model predictions were most accurate with correct species identification in most cases.

Error cases typically involved visually similar species.

Augmentation improved generalization and performance under image noise/variation.
