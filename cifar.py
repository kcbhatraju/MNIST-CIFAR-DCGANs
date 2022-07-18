from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

net = models.Sequential()
net.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
net.add(layers.MaxPooling2D((2, 2)))
net.add(layers.Conv2D(64, (3, 3), activation="relu"))
net.add(layers.MaxPooling2D((2, 2)))
net.add(layers.Conv2D(64, (3, 3), activation="relu"))
net.add(layers.Flatten())
net.add(layers.Dense(64, activation="relu"))
net.add(layers.Dense(10))

net.summary()

net.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
history = net.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = net.evaluate(test_images, test_labels,verbose=2)
print(test_acc)
