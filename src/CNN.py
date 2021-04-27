#%% Imports
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#%% Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#%% Model
model = models.Sequential()

kernels = [32, 64]
conv_window_size = (5, 5)
pooling_size = (2, 2)
number_of_classes = 10

model.add(layers.Conv2D(
    kernels[0], 
    conv_window_size,
    activation='relu',
    input_shape=(28, 28, 1))) # The 1 is there because it's going to work with B&W images
model.add(layers.MaxPooling2D(pooling_size))
model.add(layers.Conv2D(
    kernels[1],
    conv_window_size,
    activation='relu'))
model.add(layers.MaxPooling2D(pooling_size))
model.add(layers.Flatten())
model.add(layers.Dense(number_of_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
training_data = model.fit(train_images, train_labels,
          batch_size=100,
          epochs=10,
          verbose=1,
          validation_split=0.2)

#%% Results
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

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

