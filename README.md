# Fashion Product Classification using ResNet50 and TensorFlow

## Overview

This project implements a deep learning-based fashion product classification system using ResNet50 transfer learning and TensorFlow. The model is trained on over 44,000 fashion product images and classifies products into 7 major fashion categories.

## Features

* Fashion image classification using deep learning
* Transfer learning with ResNet50
* Automated image preprocessing and normalization
* Multi-class fashion category prediction
* TensorFlow data pipeline for efficient training
* Performance evaluation using validation accuracy

## Dataset

The project uses the Fashion Product Images Dataset containing:

* 44,441 fashion product images
* Product metadata stored in JSON format
* 7 master categories:

  * Apparel
  * Accessories
  * Footwear
  * Personal Care
  * Free Items
  * Sporting Goods
  * Home

## Model Architecture

* ResNet50 (ImageNet Pretrained Weights)
* Global Average Pooling
* Dense Layer (256 Units, ReLU)
* Dropout (0.3)
* Dense Output Layer (7 Classes, Softmax)

## Training Pipeline

1. Image Loading
2. Image Resizing (224 × 224)
3. Pixel Normalization
4. Label Encoding
5. Train-Test Split
6. Transfer Learning using ResNet50
7. Model Training and Evaluation

## Technologies Used

* Python
* TensorFlow / Keras
* ResNet50
* NumPy
* Scikit-learn
* Matplotlib

## Results

* Training Accuracy: 91%
* Validation Accuracy: 88%

## Future Improvements

* Fine-tune ResNet50 layers for improved performance
* Implement visual product matching
* Fashion recommendation system
* Similar image retrieval using embeddings
* Web application deployment with Streamlit

## Installation

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

## Run the Project

```bash
python train.py
```

## Project Structure

```text
Fashion-Product-Classification/
│
├── dataset/
├── notebooks/
├── models/
├── images/
├── README.md
└── requirements.txt
```

## Author

Niti Jain

## License

This project is developed for educational and portfolio purposes.
