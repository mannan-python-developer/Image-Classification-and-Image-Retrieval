# Image Classification Project

## Overview

This project focuses on image classification using deep learning techniques, particularly convolutional neural networks (CNNs), and similarity search algorithms. The main goal is to classify images into predefined categories and to find visually similar images within a dataset.

## Features

- **Preprocessing:** Images are preprocessed using TensorFlow and Keras utilities to ensure compatibility with the CNN model.
- **Model Loading:** A pre-trained CNN model, such as VGG16, is loaded using TensorFlow's Keras API. This model has been trained on large-scale image datasets and can extract high-level features from images.
- **Dataset:** The CIFAR-100 dataset is utilized for training and testing the image classification model. CIFAR-100 consists of 60,000 32x32 color images in 100 classes, with 600 images per class.
- **Similarity Search:** After extracting features from images using the CNN model, similarity search algorithms, such as cosine similarity, are applied to find visually similar images within the dataset.
- **Visualization:** Results are visualized using matplotlib, displaying the query image alongside the top-k most similar images with their corresponding class labels.

## Project Structure

- **Note:** The pre-trained CNN model file (`best_model.h5`) is not provided in this repository due to its large size (greater than 25 MB). You can train your own model or obtain a pre-trained model from other sources and if you need the model file you can contact me.
- `README.md`: Markdown file containing project description, usage instructions, and other relevant information.
- `image_classification.ipynb`: Jupyter Notebook containing the main code for image classification, including preprocessing, feature extraction, similarity search, and visualization.
- `G:\\Images\\rose.jpeg`: Example query image for similarity search.


## Usage

1. Clone the repository to your local machine.
2. Ensure you have Python installed along with the necessary libraries specified in the below section.
3. Open the `image_classification.ipynb` notebook using Jupyter Notebook or any compatible environment.
4. Execute the code cells in the notebook sequentially to load the pre-trained model, preprocess images, perform similarity search, and visualize the results.
5. Experiment with different query images and parameters to explore the capabilities of the image classification system.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- SciPy
- Matplotlib

