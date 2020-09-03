import keras
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

#import/load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

#normalize test data values between 0-1
x_test = tf.keras.utils.normalize(x_test, axis=1)

#load model
model = tf.keras.models.load_model('number_predictor.model')

predictions = model.predict(x_test[:5])


count = 0
for x in range(len(predictions)):
    guess = np.argmax(predictions[x])
    actual = y_test[x]
    print("prediction: ", guess)
    print("actual: ", actual)
    if(guess != actual):
        count+=1
    
    plt.imshow(x_test[x], cmap = plt.cm.binary)
    plt.show()

p = len(predictions)
correct_percentage = 100 *((p-count)/p)

print("the program was " + str(correct_percentage) + "%% correct")