import numpy as np
from numpy.random.mtrand import sample

np.random.seed(0)


def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y


class dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # shape of weights here is already transposed (actual shape of weights
        # usualy is number of neurons, inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    # relu function removes all the negative values and replaces them with a
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_SoftMax:
    # Exponentiate and normalize the input data math.e ^ input data
    def forward(self, input):
        exponents = np.exp(input - np.max(input, axis=1, keepdims=True))
        # substract the max from input to avoid overflowing after np.exp
        output = exponents/np.sum(exponents, axis=1, keepdims=True)
        self.output = output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# Common loss class


class Loss:
    # Calculates the data and regularization losses # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    # Number of samples in a batch
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        # Probabilities for target values - # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # convert sparse to one_hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = (- y_true / dvalues)
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_SoftMax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
    # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
    # If labels are one-hot encoded,
    # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
    # Copy so we can safely modify
        self.dinputs = dvalues.copy()
    # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
    # Normalize gradient
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0, decay=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def before_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):
        layer.weights += (-self.current_learning_rate * layer.dweights)
        layer.biases += (-self.current_learning_rate * layer.dbiases)

    def after_update_params(self):
        self.iterations += 1


X, y = spiral_data(100, 3)
layer1 = dense(2, 64)
actvation1 = Activation_ReLU()
layer2 = dense(64, 3)
lossActivationFunction = Activation_Softmax_Loss_CategoricalCrossentropy()
Optimizer = Optimizer_SGD(decay=1e-4)

for epoch in range(10000):

    layer1.forward(X)
    actvation1.forward(layer1.output)

    layer2.forward(actvation1.output)
    loss = lossActivationFunction.forward(layer2.output, y)

    predictions = np.argmax(lossActivationFunction.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f} ' + f'LR: {Optimizer.current_learning_rate}')

    # backward pass
    lossActivationFunction.backward(lossActivationFunction.output, y)
    layer2.backward(lossActivationFunction.dinputs)
    actvation1.backward(layer2.dinputs)
    layer1.backward(actvation1.dinputs)

    # Update weights and biases
    Optimizer.before_update_params()
    Optimizer.update_params(layer1)
    Optimizer.update_params(layer2)
    Optimizer.after_update_params()
