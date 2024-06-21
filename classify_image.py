# classify_image.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Function to preprocess image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))  # Resize to match model input dimensions
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict image class
def predict_image_class(model, image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    return predicted_class

# Function to display image and prediction
def display_prediction(image_path, class_names, predicted_class):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Predicted Class: {class_names[predicted_class]}')
    plt.show()

if __name__ == '__main__':
    # Path to your saved model
    model_path = 'image_classifier_model.h5'
    
    # Load the model
    model = load_trained_model(model_path)
    
    # Path to the image you want to classify
    image_path = r'D:\AIprojetcs\ImageClassifier\test images\lala.jpg'
    
    # If you have class names from your CIFAR-10 dataset
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Predict the class of the image
    predicted_class = predict_image_class(model, image_path)
    
    # Display the prediction
    display_prediction(image_path, class_names, predicted_class)
