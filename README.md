# Leaf Classification using Self-Supervised Learning (SimCLR)

A deep learning-based leaf classification system that combines Self-Supervised Learning (SimCLR) with supervised fine-tuning to classify leaf images across 9 plant species.

The project includes SimCLR-based representation learning, a custom CNN encoder, supervised classification, model evaluation, and an interactive Streamlit web application for real-time predictions.

## Live Demo

[Launch the Leaf Classifier](https://leaf-classification-nn-xteamlrhv3f5ch6rylayzb.streamlit.app/)

Upload a leaf image to get the predicted species, confidence score, and class-wise probability distribution.

## Overview

Traditional image classification requires a large amount of labeled training data. This project explores Self-Supervised Learning using SimCLR to learn meaningful visual representations before performing supervised classification.

The overall workflow is:

Leaf Images
    |
    v
Data Augmentation
    |
    v
SimCLR Self-Supervised Pretraining
    |
    v
CNN Feature Representation
    |
    v
Supervised Fine-Tuning
    |
    v
9-Class Leaf Classification
    |
    v
Streamlit Deployment

## Key Features

- Self-Supervised Learning using SimCLR
- Custom CNN Encoder
- Contrastive Representation Learning
- Supervised Fine-Tuning
- Image Augmentation and Preprocessing
- Accuracy Analysis
- Confusion Matrix Evaluation
- 9-Class Leaf Classification
- Interactive Streamlit Web Application
- Prediction Confidence and Class Probabilities
- Streamlit Community Cloud Deployment
- Git LFS for Large Model Storage

## Project Structure

Leaf-Classification-NN/
|
|-- Images/
|   |-- accuracy_plot.png
|   `-- confusion_matrix.png
|
|-- dataset/
|   `-- README.md
|
|-- models/
|   `-- README.md
|
|-- src/
|   |-- nn_pretrain.py
|   |-- rest.py
|   |-- predict.py
|   `-- app.py
|
|-- leaf_classifier.pth
|-- requirements.txt
|-- .gitignore
|-- .gitattributes
`-- README.md

## Methodology

### 1. Self-Supervised Pretraining - SimCLR

The first stage uses SimCLR (Simple Framework for Contrastive Learning of Visual Representations) to learn meaningful image representations without relying on class labels.

The process includes:

- Generating multiple augmented views of the same image
- Passing augmented images through a CNN encoder
- Generating feature representations
- Using a projection head for contrastive learning
- Maximizing similarity between representations of the same image
- Minimizing similarity between representations of different images

Original Image
    |
    v
Data Augmentation
    |
    +----------+
    |          |
    v          v
  View 1     View 2
    |          |
    +----+-----+
         |
         v
    CNN Encoder
         |
         v
 Projection Head
         |
         v
 Contrastive Loss

The resulting encoder learns useful visual features that can be transferred to the supervised classification stage.

### 2. Supervised Fine-Tuning

The pretrained encoder is used as the feature extractor for the classification task.

The classification architecture consists of:

Input Image
    |
    v
CNN Encoder
    |
    v
256-D Feature Representation
    |
    v
Linear Layer (256 -> 128)
    |
    v
ReLU
    |
    v
Dropout
    |
    v
Linear Layer (128 -> 9)
    |
    v
Class Prediction

The model is fine-tuned using labeled leaf images to classify the nine target species.

### 3. Prediction System

The prediction pipeline performs the following steps:

1. Load the trained PyTorch model.
2. Upload a leaf image.
3. Resize the image to 224 x 224.
4. Convert the image into the required tensor format.
5. Perform model inference.
6. Calculate class probabilities using Softmax.
7. Return the predicted leaf species and confidence score.

## Supported Classes

The model classifies images into the following 9 leaf species:

1. Ashok Leaves
2. Banana Leaves
3. Blackboard Leaves
4. Gulmohar Leaves
5. Jamun Leaves
6. Lily Leaves
7. Neem Leaves
8. Paper Flower Leaves
9. Sadabahar (Madagascar) Leaves

## Model Performance

### Validation Accuracy

![Validation Accuracy](Images/accuracy_plot.png)

- Final Validation Accuracy: approximately 90%
- Training performed for approximately 35 epochs
- Stable convergence observed after approximately 20 epochs
- Strong classification performance across the nine leaf classes

### Confusion Matrix

![Confusion Matrix](Images/confusion_matrix.png)

The confusion matrix shows a strong diagonal pattern, indicating that the model correctly classifies most validation samples.

Minor misclassification is observed between visually similar leaf species.

## Streamlit Web Application

The project includes an interactive Streamlit application for real-time leaf classification.

### Application Features

- Upload .jpg, .jpeg, or .png leaf images
- Display the uploaded image
- Predict the leaf species
- Display prediction confidence
- Visualize class-wise probability distribution
- Responsive wide-layout interface
- Real-time model inference

### Prediction Workflow

Upload Leaf Image
    |
    v
Image Preprocessing
    |
    v
Trained CNN Model
    |
    v
Prediction
    |
    v
Confidence Score
    |
    v
Class Probability Distribution

### Live Application

https://leaf-classification-nn-xteamlrhv3f5ch6rylayzb.streamlit.app/

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Akash7644/Leaf-Classification-NN.git
cd Leaf-Classification-NN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Train the Model

```bash
python src/rest.py
```

### Run Prediction

```bash
python src/predict.py
```

### Run the Streamlit Application

```bash
streamlit run src/app.py
```

## Model Management

The trained classification model is:

leaf_classifier.pth

The model is approximately 100 MB, so it is managed using Git Large File Storage (Git LFS) instead of standard Git storage.

To work with the model locally:

```bash
git lfs install
git lfs pull
```

The Streamlit application automatically loads the model from the repository root.

## Technologies Used

### Programming
- Python

### Deep Learning
- PyTorch
- Torchvision
- SimCLR
- Convolutional Neural Networks

### Machine Learning
- Scikit-learn

### Data Visualization
- Matplotlib
- Seaborn

### Image Processing
- Pillow

### Deployment and Version Control
- Streamlit
- Streamlit Community Cloud
- Git
- GitHub
- Git LFS

## Key Insights

- Self-supervised pretraining enables learning useful visual representations before supervised training.
- Contrastive learning helps the encoder learn meaningful visual features from augmented image pairs.
- Fine-tuning the pretrained encoder adapts the learned representations to leaf classification.
- Data augmentation improves the model's ability to generalize to variations in leaf images.
- The final model achieves approximately 90% validation accuracy across nine leaf classes.
- Streamlit provides an accessible interface for real-time model inference.

## Future Improvements

- Experiment with pretrained architectures such as ResNet and EfficientNet.
- Perform systematic hyperparameter optimization.
- Increase dataset size and class diversity.
- Improve classification of visually similar leaf species.
- Add Grad-CAM for model interpretability.
- Add Top-3 prediction results.
- Add real-time camera-based leaf classification.
- Optimize inference performance for cloud deployment.

## Author

Akash Badgoti

Artificial Intelligence & Data Science
MBM University, Jodhpur

### Project Links

- GitHub: https://github.com/Akash7644/Leaf-Classification-NN
- Live Demo: https://leaf-classification-nn-xteamlrhv3f5ch6rylayzb.streamlit.app/

## Support

If you found this project useful, consider giving the repository a star on GitHub.
