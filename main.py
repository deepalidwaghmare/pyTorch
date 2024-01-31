import tensorflow as tf
import matplotlib.pyplot as plt

# get the data and split into Train_Data and Test_Data

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

print(train_images.shape)
print(test_images.shape)
print(train_labels)

plt.imshow(train_images[0], cmap = 'gray')
plt.show()

# define neural network
my_model = tf.keras.models.Sequential()
my_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
my_model.add(tf.keras.layers.Dense(128, activation='relu'))  # why Relu
my_model.add(tf.keras.layers.Dense(10,
                                   activation='softmax'))  # softmax to convert values into probablistic values which is easy to compare

# compile model
my_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])  # adam is the best optimisation algo in neural network

# train model
my_model.fit(train_images, train_labels, epochs=3)

# test model
val_loss, val_acc= my_model.evaluate(test_images, test_labels)
print('Test Accuracy:', val_acc)
