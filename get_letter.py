import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

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
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

def display_results(img, predicted_class):
    # Display results
    plt.imshow(img)
    plt.title(f'Predicted ASL Sign: {predicted_class}')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_letter.py FILE_NAME.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    model = load_model()
    img, img_array = preprocess_image(img_path)
    predicted_class = predict_letter(model, img_array)
    display_results(img, predicted_class)

if __name__ == "__main__":
    main()