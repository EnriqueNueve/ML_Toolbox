"""
grad_tape_demo.py
A low level example of a Neural Network in Tensorflow
using tf.Variable and tf.GradientTape().
"""

# Import modules
import tensorflow as tf
import numpy as np

#--------------------------#

# Declare optimizer function
opt = tf.optimizers.Adam(learning_rate=.001, decay = 1e-6)

# Declare loss function
def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

# Declare class called Model to hold tf.Variables and to perform a foward pass 
class Model(object):
    def __init__(self):
        self.w1 = tf.Variable([[0.2,-0.8,-0.5,0.6],[-0.5,0.6,1.2,0.3]],dtype=tf.float32)
        self.b1 = tf.Variable(0.0)
        self.w2 = tf.Variable([[0.9,0.7],[1.0,1.0],[0.9,0.7],[.2,1]],dtype=tf.float32)
        self.b2 = tf.Variable(0.0)

    def __call__(self,x):
        x1 = tf.matmul(self.w1,x)
        x2 = tf.keras.activations.sigmoid(x1+self.b1)
        x3 = tf.matmul(self.w2,x2)
        yh = tf.keras.activations.softmax(x3+self.b2,axis=-1)
        return yh

#--------------------------#

# Create data set, XOR
X = tf.constant([[.01,.01],[.01,.99],[.99,.99],[.99,.01]],dtype=tf.float32)
Y = tf.constant([[1.0,0.0],[0.0,1.0],[1.0,0.0],[0.0,1.0]],dtype=tf.float32)

#--------------------------#

# Declare model
model = Model()

# Train weights
def train(model, inputs, outputs,learning_rate = 0.1):
    with tf.GradientTape() as tape:
        current_loss = loss(outputs, model(inputs))
    dw1, dw2, db1, db2 = tape.gradient(current_loss, [model.w1, model.w2,model.b1,model.b2])
    model.w1.assign_sub(learning_rate * dw1)
    model.w2.assign_sub(learning_rate * dw2)
    model.b1.assign_sub(learning_rate * db1)
    model.b2.assign_sub(learning_rate * db2)

for i in range(1000):
    train(model,X,Y)

#--------------------------#

# Test model
results = model(X)
print("Prediction: ",np.argmax(results.numpy(),axis=-1))
print("Actual: ",np.argmax(Y.numpy(),axis=-1))
