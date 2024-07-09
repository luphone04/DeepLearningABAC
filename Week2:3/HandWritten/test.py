import tensorflow as tf
from keras.models import load_model


# Load the model
model = load_model('/Users/richard/Desktop/DeepLearningABAC/Week2/HandWritten/mnist.h5')

# Add this code after loading the model
print(model.summary())
