# neural-network-v1
Based on 3Blue1Brown (https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

This neural network comes neatly packaged in few functions and classes, so it can be easily added to any preexisting code or program. Many parameters, such as the number/size of layers in the network, are fully customizable.

This code will include:
  - A NeuralNetwork object, from which inputs and outputs to/from the network can be made
  - Node/Neuron objects, which will contain the data of a single node/neuron in the network
  - Weight objects, which will connect the nodes from adjacent layers, and store the data of a weight in the network 

This could change, but I believe a set of training data (AKA the inputs and correct outputs) will be required upon creating the NeuralNetwork object, and stored within the object as needed. The data itself can be changed later, but not its size, as this would affect the number of nodes and weights in the network.

To train the algorithm:
  - Take each input from the training data, and generate a result of output neurons, in addition to its one-answer output
  - Backpropagate the desired changes to the output neurons
  - Repeat
