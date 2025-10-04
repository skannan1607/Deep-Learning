# SIGMOID FUNCTION

import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function :
def sigmoid(x):
 return 1/(1+np.exp(-x))

ip = -1
op = sigmoid(ip)
print(op)

# Define a range of x-values
x = np.linspace(-10, 10, 1000) # 1000 points between -10 and 10

# Calculate the corresponding y-values using the sigmoid function
y = sigmoid(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.title("Sigmoid Function Graph")
plt.grid(True)
plt.show()

# TANH ACTIVATION

# Tanh function :
def tanh(x):
 return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# Calculate the corresponding y-values using the tanh function
y = tanh(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("tanh(x)")
plt.title("Tanh Function Graph")
plt.grid(True)
plt.show()

# RELU ACTIVATION

# ReLU function :
def relu(x):
  return np.maximum(0, x)

# Calculate the corresponding y-values using the ReLU function
y = relu(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("relu(x)")
plt.title("ReLU Function Graph")
plt.grid(True)
plt.show()

# SOFTMAX

# softmax function
def softmax(x):
  a = np.exp(x)
  a = a/np.sum(a)
  return a

# Calculate the corresponding y-values using the ReLU function
y = softmax(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("soft(x)")
plt.title("Softmax Function Graph")
plt.grid(True)
plt.show()

# LINEAR ACTIVATION

def linear(x):
  return x

# Calculate the corresponding y-values using the ReLU function
y = linear(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("linear(x)")
plt.title("Linear Function Graph")
plt.grid(True)
plt.show()

# STEP ACTIVATION

# Sample input data
x = np.linspace(-5, 5, 100)

def step_function(x):
    return np.where(x >= 0, 1, 0)

# Calculate outputs
y = step_function(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("step(x)")
plt.title("Step Function Graph")
plt.grid(True)
plt.show()

# Signum activation function

# Sample input data
x = np.linspace(-5, 5, 100)

def signum_function(x):
    return np.where(x > 0, 1, np.where(x < 0, -1, 0))

y = signum_function(x)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("Signum(x)")
plt.title("signum Function Graph")
plt.grid(True)
plt.show()
