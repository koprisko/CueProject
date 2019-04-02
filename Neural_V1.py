from numpy import exp, array, random, dot
import LoadCSV_V2
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((4, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return ((1 / (1 + exp(-x))))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return (x * (1 - x))

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.synaptic_weights)

    # The training set. We are unzipping the dataset and putting it into an array
    training_set_inputs = LoadCSV_V2.normal_input("MSFT")
    temp_list = []
    normal_close = []
    normal_high = []
    normal_low = []
    normal_prev = []
    unzip = list(zip(*(training_set_inputs)))
    normal_close = unzip[0]
    normal_high = unzip[1]
    normal_low = unzip[2]
    normal_prev = unzip[3]
    for i in range(len(training_set_inputs)):
        temp_list = [normal_close[i],normal_high[i],normal_low[i],normal_prev[i]]
        training_set_inputs[i] = temp_list
    
    
    #Select 70% of the data set for the training set
    train_end = int(.7*len(training_set_inputs))
        
    training_set_inputs = array(training_set_inputs)
    training_set_outputs = array([LoadCSV_V2.normal_output("MSFT")]).T
    
    training_set_inputs_new = training_set_inputs[0:train_end]
    training_set_outputs_new = training_set_outputs[0:train_end]    
    test_set_inputs = training_set_inputs[train_end+1:]
    test_set_outputs = training_set_outputs[train_end+1:]
    
    
    
    
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
   # print(training_set_inputs)
    #print(training_set_outputs)
    neural_network.train(training_set_inputs, training_set_outputs, 100)

    print ("New synaptic weights after training: ")
    print (neural_network.synaptic_weights)

    # Test the neural network with a new situation.
    print ("Considering new situation -> ?: ")
    guess = (neural_network.think(array(test_set_inputs[0])))
    guess = LoadCSV_V2.reverse_output("MSFT",guess)
    print (guess)
    print ("Actual: ")
    actual = test_set_outputs[0]
    actual = LoadCSV_V2.reverse_output("MSFT",actual)
    print (actual)
    
    # Store predicted data in lists 
    results = []
    amount = []
    for n in range(len(test_set_outputs)):
        results.append(neural_network.think(array(test_set_inputs[n])))
        amount.append(n)
    
        
    # Graph the actual data vs the predicted data
    
    plt.plot(amount, results, label = "predicted")
    plt.plot(amount, test_set_outputs, label = "actual")
    plt.legend()
    plt.show()
    
