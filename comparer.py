import cv2
import tensorflow as tf


def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


prediction_name = ["Banana", "Lemon", "Orange"]
model = tf.keras.models.load_model(
    "") #Path to trained model file

prediction = model.predict([prepare('')]) #Path to photo of desired object to identify
print("Prediction Banana =", prediction[0][0])
print("Prediction Lemon =", prediction[0][1])
print("Prediction Orange =", prediction[0][2])
print("Prediction Final =", prediction_name[max(range(len(prediction[0])), key=prediction[0].__getitem__)])
