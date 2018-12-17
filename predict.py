import codecs
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf


"""Learning parameters"""
learning_rate = 0.1
training_epochs = 5
display_step = 1

x_train, y_train, z_train = [], [], []

"""Read data from json"""
with codecs.open("authors.json", "r", encoding="UTF-8") as f:
    authors = json.loads(f.read())

for author in authors:
    hindex = int(author["hindex"])
    ndocuments = int(author["ndocuments"])
    citation_count = int(author["citation_count"])

    x_train.append([ndocuments])
    y_train.append([citation_count])
    z_train.append([hindex])

"""Normalize data"""
x_original = x_train

x_train = np.array(x_train)
x_train = tf.keras.utils.normalize(x_train, axis=0)

y_original = y_train

y_train = np.array(y_train)
y_train = tf.keras.utils.normalize(y_train, axis=0)

z_train = np.array(z_train)
z_train = tf.keras.utils.normalize(z_train, axis=0)


n_samples = x_train.shape[0]

"""Plot data before learning"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Documents')
ax.set_ylabel('Citations')
ax.set_zlabel('H-index')
ax.scatter(x_train, y_train, z_train, c="r", marker="o", label="Actual")

"""Hypothesis: Z = X * w1 + Y * w2 + b """
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(np.random.randn(), name="weight1")
W2 = tf.Variable(np.random.randn(), name="weight2")

b = tf.Variable(np.random.randn(), name="bias")

Z = tf.placeholder(tf.float32)

prediction = tf.add(tf.add(tf.multiply(X, W1), tf.multiply(Y, W2)), b)

cost_history = np.empty(shape=[1], dtype=float)

# Mean squared error
cost = tf.reduce_sum(tf.pow(prediction - Z, 2)) / (2 * n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

# Fit all training data
for epoch in range(training_epochs):
    for (x, y, z) in zip(x_train, y_train, z_train):
        d = {X: x, Y: y, Z: z}
        sess.run(optimizer, feed_dict=d)

        cost_history = np.append(cost_history, sess.run(cost, feed_dict=d))

    if (epoch + 1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_history[-1]))

ax.scatter(x_train, y_train, x_train * sess.run(W1) + y_train * sess.run(W2) + sess.run(b), "gray", label="Predicted")
ax.legend()
plt.show()
