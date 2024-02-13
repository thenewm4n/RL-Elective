import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot

# Accessing MNIST dataset with tensorflow
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images, test_images = training_images / 255.0, test_images / 255.0

def relu(x):
    return np.maximum(0, x)

def display_images(images, labels, num_images = 10):
    figure, axes = plot.subplots(1, num_images, figsize = (20, 2)) # 1 row, num_images number of columns
    for i in range(num_images):
        axes[i].imshow(images[i], cmap = 'gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plot.tight_layout()
    plot.show()

class MultilayerPerceptron():
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(layer_sizes[i + 1], layer_sizes[i]) for i in range(len(layer_sizes) - 1)]    # creates list of matrices
        self.biases = [np.random.rand(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]                      # i+1 because biases needed only for layers after input layer
    
    def forward_propagate(self, input_activations):                         # input_activations is a vector
        activations = [input_activations]                                   # initialises the input activations as first element in a list of activations for each layer
        for i in range(len(self.layer_sizes) - 1):                 
            activations.append(relu(np.dot(self.weights[i], activations[i]) + self.biases[i]))      # ideally for readability, activations index would be 1 less than weights and biases; relu applied to the product of the weights 
        return activations
    
    def calculate_cost(self, actual_activations, desired_activations):          # Calculates cost of one image
        cost = np.sum(np.square(actual_activations[len(self.layer_sizes) - 1] - desired_activations[len(self.layer_sizes) - 1]))

    def back_propagate(self, actual_activations, desired_activations):          # Calculates negative gradient
        
    # def adjust_parameters(self, gradients, learning_rate):       # Shift weights and biases according to gradient



LAYER_SIZES = [784, 16, 16, 10]
BATCH_SIZE = 100
mlp = MultilayerPerceptron(LAYER_SIZES)


# for i in range(BATCH_SIZE):
#     activations = mlp.forward_propagate(training_images[i])
#     cost = mlp.calculate_cost(activations, training_labels[i])

# display_images(training_images, training_labels)
# print(activations[len(LAYER_SIZES) - 1])