
# Machine Learning Assignments

This repository contains two assignments from the "Machine Learning" graduate course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics - Biomedical Data Science Specialization) of the Department of Informatics and Telecommunications department of the National and Kapodistrian University of Athens (NKUA), under the supervision of professor Stavros Perantonis, in the academic year 2022-2023. These assignments cover key concepts such as regression, classification, adversarial examples, and neural networks. Each assignment provides practical implementations of machine learning algorithms and frameworks, using datasets such as IRIS PLANT and MNIST.

---

## Contributors

- [Konstantinos Giatras](https://github.com/GiatrasKon)
- [Despina Konstantinidou](https://github.com/dekonstantinidou)
- [Isidoros Stamatiou](https://github.com/NuncioQ)
- [Christos Vasilopoulos](https://github.com/Randabas)

## Assignment 1: Regression and Classification Analysis

### Overview

The first assignment focuses on implementing regression methods and k-Nearest Neighbors (k-NN) classifiers. Key methods include Least Squares Regression, Ridge Regression, Bayesian Inference, and k-NN-based classification applied to datasets like the Iris Plant and Pima Indians Diabetes databases.

### Main Workflow

1. Regression:
    - Implemented Least Squares Regression, Ridge Regression, and Bayesian Inference.
    - Compared MSE for different regression models.
    - Explored the bias-variance tradeoff using 2nd and 10th-degree polynomials.
2. k-NN Classification:
    - Built a k-NN classifier to classify Iris plant species and detect diabetes in the Pima Indians dataset.
    - Analyzed performance metrics with varying numbers of neighbors and cross-validation.

### Key Results

- Demonstrated the effect of regularization (Ridge Regression) on reducing test MSE.
- Observed overfitting in higher-degree polynomial regression.
- Achieved significant classification accuracy improvements using optimized k values for the k-NN classifier.

---

## Assignment 2: Deep Learning and Adversarial Robustness

### Overview
The second assignment delves into neural networks, adversarial examples, and real-time emotion recognition using datasets like MNIST, CIFAR-10, and Kaggle's Emotion Dataset.

### Main Workflow
1. Neural Networks:
    - Trained fully connected and convolutional neural networks on MNIST.
    - Explored the impact of activation functions (ReLU, Sigmoid, etc.) and network depth on performance.
    - Visualized gradients for deeper insights into model behavior.
2. Adversarial Examples:
    - Generated adversarial images for MNIST and CIFAR-10 datasets.
    - Examined the vulnerabilities of neural networks to adversarial attacks and proposed robust models.
3. Emotion Recognition:
    - Built a real-time emotion recognition classifier capable of predicting emotions using a laptop camera feed.
    - Model trained on Kaggle's facial emotion recognition dataset, saved in .h5 format for deployment.

### Key Results
- Achieved competitive test scores across various neural network architectures for MNIST and CIFAR-10.
- Successfully generated adversarial examples showcasing model vulnerabilities.
- Implemented real-time emotion detection with significant accuracy across multiple emotions.

---

## Installation

### Cloning the Repository

```sh
git clone https://github.com/GiatrasKon/Machine_Learning_Assignments.git
```

### Datasets Availability

1. [**Iris Plant Database**](https://www.kaggle.com/datasets/arshid/iris-flower-dataset): Dataset for classifying three types of Iris plants based on sepal and petal dimensions.
2. [**Pima Indians Diabetes Database**](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database): Dataset for predicting diabetes status in Pima Indian women based on medical measurements.
3. **MNIST (Modified National Institute of Standards and Technology Database)**: A dataset of handwritten digits (0-9) with 28x28 grayscale images.
    - Can be directly loaded via `keras.datasets.mnist`.
4. **CIFAR-10**: A dataset containing 60,000 32x32 color images across 10 classes.
    - Can be directly loaded via `keras.datasets.cifar10`.
5. [**Facial Emotion Recognition Dataset**](https://www.kaggle.com/code/gauravsharma99/facial-emotion-recognition/input): A dataset for emotion classification including images labeled with seven emotion categories.

### Package Dependencies

Assignment 1:
- numpy
- math
- random
- pandas
- matplotlib
- scikit-learn
- scipy

Assignment 2:
- tensorflow
- tensorflow_datasets
- keras
- matplotlib
- numpy
- pandas
- scikit-learn
- tqdm
- IPython.display
- opencv-python
- base64

You can install all the dependencies at once using the following command:

```sh
pip install numpy math random pandas matplotlib scikit-learn scipy tensorflow tensorflow-datasets keras tqdm ipython opencv-python base64
```

---

## Usage Instructions

1. Open the relevant Jupyter Notebook (`.ipynb`) file in the corresponding assignment directory.
2. Execute the code cells step-by-step for full functionality.
3. To run the real-time emotion recognition (Assignment 2, Exercise 3):
    - Ensure you have a functional camera device.
    - Run the provided Python script to classify emotions in real-time.
    - **NOTE**: The implementation of this exercise only works in google colab.

---