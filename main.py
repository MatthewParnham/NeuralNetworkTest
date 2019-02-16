import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense

np.random.seed(0)

(X_train, y_train), (X_test, y_test) = mnist.load_data() #Split Data into test/train split

plt.figure(figsize=(15,10)) # Display w/ matplotlib
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))


num_classes = 10

X_train = X_train.astype('float32').reshape(-1,784)
X_test = X_test.astype('float32').reshape(-1,784)

X_train /= 255.; X_test /= 255.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Building model
model = Sequential()
model.add(Dense(256, activation='relu',input_shape=(784,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
# Train model
model.fit(X_train,y_train,
         batch_size=128,
         epochs=40)


# Make predictions and roughly check them
y_pred = model.predict(X_test)
for i in range(20):
    print("Prediction: " + str(y_pred[i]) + "\nAnswer: " + str(y_test[i]))
