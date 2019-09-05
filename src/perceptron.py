"""
Neural Perceptron Network

@author Caio Tomich (caiotomich@gmail.com)
@version 1.0.0
@since 2019-08-27;
"""

import numpy as np
import matplotlib.pyplot as plt


def activation_func(val):
    if val < 0:
        return -1
    return 1


training_data = np.genfromtxt('../docs/training_data.txt', skip_header=False)

t = np.full(30, -1)
w = np.random.uniform(low=-1, high=1, size=4)
x = np.hstack((t[:, np.newaxis], training_data[:, 0:3]))
d = training_data[:, 3]

print("\n --- Training Script --- \n")
print("Initial Weight:", w)

epoch = 0
error = True
weight = []
learn_rate = 0.01

while error:
    error = False

    for i in range(len(x)):
        u = np.dot(weight, x[i])
        y = activation_func(u)

        if y != d[i]:
            w = w + (learn_rate * (d[i] - y) * x[i])
            error = True

    weight.append(w)
    epoch += 1

print("End Weight:", w)
print("Epoch:", epoch)

plt.plot(np.arange(epoch), np.array(weight))
plt.show()


print("\n --- Test Script --- \n")

validation_data = np.genfromtxt('docs/validation_data.txt', skip_header=False)

w_after_training = weight
t = np.full(len(validation_data), -1)
x = np.hstack((t[:, np.newaxis], validation_data))

y_expected = [-1, 1, 1, 1, 1, 1, -1, 1, -1, -1]
y_response = []

for i in range(len(x)):
    u = np.dot(w_after_training, x[i])
    y = activation_func(u)
    y_response.append(y)

    if y == -1:
        print("Sample: {} => Class P1".format(x[i]))
    else:
        print("Sample: {} => Class P2".format(x[i]))

print("\nExpected output after training")
print("y_expected == y_response: {}".format(np.equal(y_expected, y_response)))
