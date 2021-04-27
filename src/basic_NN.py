#%% Imports
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt

#%% Data loading
# Train and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#%% Data normalization
# Transform the data into float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Transform the numbers in numbers between 0 and 1
x_train /= 255
x_test /= 255


# Transform features shape into 1D (28 *28 = 784)
print(x_train.shape)
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784)

# Run one-hot encoding for 10 classes that represent the numbers
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#%% Model and training
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'sgd', # stocastic gradient descent
              metrics = ['accuracy'])

training_data = model.fit(x_train,
                          y_train,
                          batch_size=100,
                          epochs=40,
                          validation_split=0.2)

#%% Test accuracy
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(training_data.history['accuracy'])
plt.plot(training_data.history['val_accuracy'])
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(training_data.history['loss'])
plt.plot(training_data.history['val_loss'])
plt.legend(['Train', 'Validation'], loc = 'upper left')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

print("Real label of the number: ", np.argmax(y_test[7]))
predictions = model.predict(x_test)
print("Predicted label by the model: ", np.argmax(predictions[7]))


