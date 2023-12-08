import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import string
import cv2
import socket

def load_model():
    # Load your ASL model
    model = tf.keras.models.load_model('model.h5')
    return model

def preprocess_image(img_path):
    # Preprocess the input image
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

def predict_letter(model, img_array):
    # Make predictions
    image = cv2.imread(img_path)
    new = cv2.rotate(image, rotateCode=0)
    gimage = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    image28 = cv2.resize(gimage, (28, 28))
    imagere = image28.reshape(1, 28, 28, 1)
    onehot = model.predict_classes(imagere)
    return output[onehot[0]]

def display_results(img, predicted_class):
    # Display results
    plt.imshow(img)
    plt.title(f'Predicted ASL Sign: {predicted_class}')
    plt.show()

def map_to_actual_letter(predicted_class):
    # Map predicted class to actual letter (assuming 0 = 'a', 1 = 'b', etc.)
    actual_letter = string.ascii_lowercase[predicted_class]
    return actual_letter

def main():
    if len(sys.argv) != 3:
        print("Usage: python get_letter.py FILE_NAME.jpg output.txt")
        sys.exit(1)

    img_path = sys.argv[1]
    output_file = sys.argv[2]
    
    model = load_model()
    img, img_array = preprocess_image(img_path)
    predicted_class = predict_letter(model, img_array)
    display_results(img, predicted_class)

    # Map predicted class to actual letter
    actual_letter = map_to_actual_letter(predicted_class)

    # Append and save the result file
    with open(output_file, 'a') as f:
        f.write(f'{actual_letter}')

if __name__ == "__main__":
    main()
