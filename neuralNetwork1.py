class Weight:
    def __init__(self, startNode, endNode):
        self.startNode = startNode
        self.endNode = endNode
        self.weight = 0
    
    def setWeight(self, value):
        self.weight = value

class Node:
    def __init__(self, layerType, startValue):
        self.layerType = layerType
        self.value = startValue
        if layerType=="input":
            pass
        elif layerType=="output":
            pass
        elif layerType=="hidden":
            pass
    
    def setWeights(self, *args):
        if layerType=="input" or layerType=="hidden":
            self.outputWeights = args[-1]
        
        if layerType=="hidden" or layerType=="output":
            self.inputWeights = args[0]

class NeuralNetwork:
    def __init__(self):
        pass

    def setInputs(self, newInputs):
        self.inputs = newInputs

network = NeuralNetwork()
network.test2()