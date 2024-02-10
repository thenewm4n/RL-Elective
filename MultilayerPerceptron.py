import numpy as np

NUM_NODES_HIDDEN_LAYER = 16
NUM_NODES_INPUT_LAYER = 784
NUM_NODES_OUTPUT_LAYER = 10
NUM_HIDDEN_LAYERS = 2


# Implement forward feeding of multilayer perceptron
    # Activation function (for each node, ReLU of the sum of the product of each weight and activation of its associated previous node, and the node's bias)

input_activations = np.zeros(NUM_NODES_INPUT_LAYER)        # This is where training or testing data is input
activations_layer_0 = np.zeros(NUM_NODES_HIDDEN_LAYER)
activations_layer_1 = np.zeros(NUM_NODES_HIDDEN_LAYER)
output_activations = np.zeros(NUM_NODES_OUTPUT_LAYER)

weights_layer_0 = np.random.rand(NUM_NODES_INPUT_LAYER, NUM_NODES_HIDDEN_LAYER)
weights_layer_1 = np.random.rand(NUM_NODES_HIDDEN_LAYER, NUM_NODES_HIDDEN_LAYER)
weights_layer_2 = np.random.rand(NUM_NODES_HIDDEN_LAYER, NUM_NODES_OUTPUT_LAYER)

biases_layer_0 = np.random.rand(NUM_NODES_HIDDEN_LAYER)
biases_layer_1 = np.random.rand(NUM_NODES_HIDDEN_LAYER)     # is this correct?
biases_layer_2 = np.random.rand(NUM_NODES_OUTPUT_LAYER)

for j in range(NUM_HIDDEN_LAYERS + 1):
        for i in range(NUM_NODES_HIDDEN_LAYER):
            print("Hello, World!")


def feed_forward(layer,):
    for j in range(NUM_NODES_HIDDEN_LAYER):      
        for i in range(NUM_NODES_INPUT_LAYER):
             sum += weights_layer_0[i, j] * inputs[i]


# Implement training i.e. adjusting of weights and biases
    # Calculate cost function
    # Calculate negative gradient
    # Shift weights and biases according to gradient