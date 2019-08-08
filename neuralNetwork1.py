#if order matters in the list of weights within a node, you're gonna have to redo some stuff
import random
import math

usedIDs = []
def createNewID():
    global usedIDs
    if len(usedIDs)==999999:
        print("MAX IDS USED")
        print(1/0)
    else:
        a = random.randint(1, 1000000)
        while a in usedIDs:
            a = random.randint(1, 1000000)
        usedIDs.append(a)
        return a

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Weight:
    def __init__(self, startNode, endNode, startValue):
        self.startNode = startNode
        self.startNode.addOutput(self)
        self.endNode = endNode
        self.endNode.addInput(self)
        self.value = startValue
        self.ID = createNewID()

class Node:
    def __init__(self, layerType, startValue, startBias):
        self.layerType = layerType
        self.value = startValue
        self.bias = startBias
        self.inputWeights = []
        self.outputWeights = []
        self.ID = createNewID()
        self.error = 0
        self.delta = 0
    
    def addInput(self, weight):
        self.inputWeights.append(weight)
        
    def addOutput(self, weight):
        self.outputWeights.append(weight)

class NeuralNetwork:
    def __init__(self, shape, learningRate):
        self.layerSizes = shape
        self.createNodes()
        self.createWeights()
        self.trainingData = []
        self.learningRate = learningRate

    def clearTrainingData(self):
        self.trainingData = []

    def addTrainingData(self, newData):
        self.trainingData += newData

    def setInputs(self, newInputs):
        if len(newInputs)==self.layerSizes[0]:
            for index in range(self.layerSizes[0]):
                input = newInputs[index]
                node = self.nodeLayers[0][index]
                node.value = input
        else:
            print("NEW INPUTS OF LENGTH", len(newInputs), "DO NOT MATCH INPUT NODE COUNT", self.layerSizes[0])

    def setCorrectOutputs(self, newOutputs):
        if len(newOutputs)==self.layerSizes[-1]:
            self.correctOutputs = newOutputs
        else:
            print("NEW OUTPUTS OF LENGTH", len(newInputs), "DO NOT MATCH OUTPUT NODE COUNT", self.layerSizes[0])

    def createNodes(self):
        self.nodeLayers = []
        for index in range(len(self.layerSizes)):
            self.nodeLayers.append([])
            size = self.layerSizes[index]
            nodeType = "hidden"
            if index==0:
                nodeType=="input"
            elif index==len(self.layerSizes)-1:
                nodeType=="output"
            for a in range(size):
                self.nodeLayers[-1].append(Node(nodeType, 0, random.uniform(-1, 1)))
    
    def createWeights(self):
        self.weightLayers = []
        for index in range(len(self.layerSizes)-1):
            self.weightLayers.append([])
            for startNode in self.nodeLayers[index]:
                for endNode in self.nodeLayers[index+1]:
                    self.weightLayers[-1].append(Weight(startNode, endNode, random.uniform(-1, 1)))
    
    def feedForwardLayer(self, startIndex):
        for endNode in self.nodeLayers[startIndex+1]:
            newValue = 0
            for weight in endNode.inputWeights:
                newValue += weight.startNode.value * weight.value
            newValue += endNode.bias
            endNode.value = sigmoid(newValue)
    
    def resetNodes(self):
        for layer in self.nodeLayers: #[1:]
            for node in layer:
                node.value = 0
    
    def createOutputList(self):
        return [node.value for node in self.nodeLayers[-1]]
    
    def runExample(self):
        for a in range(len(self.weightLayers)):
            self.feedForwardLayer(a)
        return self.createOutputList()
    
    def generateOutputError(self):
        totalError = [0]*self.layerSizes[-1]
        for data in self.trainingData:
            self.setInputs(data[0])
            self.setCorrectOutputs(data[1])
            outputs = self.runExample()
            #print(outputs)
            error = [0.5*(outputs[index]-data[1][index])**2 for index in range(self.layerSizes[-1])]
            #print(error)
            totalError = [totalError[index] + error[index] for index in range(self.layerSizes[-1])]
        totalError = [totalError[index] / len(self.trainingData) for index in range(len(totalError))]
        for index in range(len(totalError)):
            outputNode = self.nodeLayers[-1][index]
            outputNode.error = outputs[index]-data[1][index]
            outputNode.delta = outputNode.value * (1-outputNode.value) * outputNode.error #sigmoid specific
        return totalError
    
    def generateHiddenError(self, layerIndex):
        for index in range(self.layerSizes[layerIndex]):
            startNode = self.nodeLayers[layerIndex][index]
            startNode.error = sum([weight.value*weight.endNode.delta for weight in startNode.outputWeights])
            startNode.delta = startNode.value * (1-startNode.value) * startNode.error
            print(startNode.delta)
    
    def updateWeightsAndBiases(self):
        for layer in self.weightLayers:
            for weight in layer:
                weight.value -= weight.endNode.delta * weight.startNode.value * self.learningRate
                #print(weight.value)
                #pass
        for layer in self.nodeLayers[1:]:
            for node in layer:
                node.bias -= node.delta * self.learningRate
                #print(node.bias)
                #pass
    
    def train(self):
        error = self.generateOutputError()
        #print("Error:", error)
        for index in range(len(self.layerSizes)-1):
            self.generateHiddenError(index)
        self.updateWeightsAndBiases()
        self.resetNodes()
    
networkShape = [2, 3, 1]
trainingData = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]
network = NeuralNetwork(networkShape, 0.5)
network.addTrainingData(trainingData)
while True:
    network.train()
    print()