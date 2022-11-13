from tensorflow import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizer_v1 import Adam
from keras.utils import np_utils
import matplotlib.pyplot as plt


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('*'*20)
print(X_train.shape)
print(X_test.shape)
print('*'*20)

print(X_train[0])
print('===============')
print(Y_train)
print('-------------')

plt.imshow(X_train[0])

X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test  = X_test.reshape(10000,28,28,1).astype('float32')

no_classes=10
Y_train = np_utils.to_categorical(Y_train,no_classes)
Y_test = np_utils.to_categorical(Y_test,no_classes)

print('===========')
print(Y_train[0])


model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(32,(3,3), activation='relu'))

model.add(Flatten())

model.add(Dense(no_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=5, batch_size=32)

metrics= model.evaluate(X_test,Y_test,verbose=0)

print("Metrics")
print(metrics)

prediction=model.predict(X_test[:4])
print(prediction)

print(np.argmax(prediction,axis=1))
print(Y_test[:4])

model.save('digit.h5')