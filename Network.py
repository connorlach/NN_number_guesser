import keras
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

#import 28x28 data 
mnist = tf.keras.datasets.mnist  

#x_train/test = data sets, y_train/test = labels
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')

#normalize data values - between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#build the model - Sequential model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten()) #input
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #hidden 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) #hidden
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) #output

#compile
model.compile(
            optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            )
#train 
model.fit(x_train, y_train, epochs=10)


#model.save('MnistModel.h5')
