#if order matters in the list of weights within a node, you're gonna have to redo some stuff
import random
import math

networkShape = [2, 3, 1] #number of nodes per layer
trainingData = [[[0, 0], [0]], [[0, 1], [0.5]], [[1, 0], [1]], [[1, 1], [1]]] #each data point is [[inputs], [outputs]]
learningRate = 0.5 #rate of weight changes

usedIDs = []
def createNewID():
    global usedIDs
    if len(usedIDs)==999999:
        raise Exception("MAX IDS USED")
    else:
        a = random.randint(1, 1000000)
        while a in usedIDs:
            a = random.randint(1, 1000000)
        usedIDs.append(a)
        return a

def sigmoid(x):
    if x<0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

class Weight:
    def __init__(self, startNode, endNode, startValue):
        self.startNode = startNode
        self.startNode.addOutput(self)
        self.endNode = endNode
        self.endNode.addInput(self)
        self.value = startValue
        self.ID = createNewID()
        self.change = 0

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
        self.biasChange = 0
    
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
    
    def train(self):
        totalError = [0]*self.layerSizes[-1]
        for data in self.trainingData:
            self.setInputs(data[0])
            self.setCorrectOutputs(data[1])
            outputs = self.runExample()
            #for index in range(self.layerSizes[-1]):
            #    print("Expected", data[1][index], "| Got", outputs[index])
            displayError = [0.5*(outputs[index]-data[1][index])**2 for index in range(self.layerSizes[-1])]
            trueError = [outputs[index]-data[1][index] for index in range(self.layerSizes[-1])]
            #print(error)
            totalError = [totalError[index] + displayError[index] for index in range(self.layerSizes[-1])]
            for index in range(self.layerSizes[-1]):
                outputNode = self.nodeLayers[-1][index]
                outputNode.dEdO = trueError[index]
            self.changeWeightsAndBiases()
        totalError = [totalError[index] / len(self.trainingData) for index in range(len(totalError))]
        for layer in self.weightLayers:
            for weight in layer:
                weight.value += weight.change
                weight.change = 0
        for layer in self.nodeLayers[1:]:
            for node in layer:
                node.bias += node.biasChange
                node.biasChange -= 0
        self.resetNodes()
        return sum(totalError)
    
    def calculatedOdZ(self):
        for layer in self.nodeLayers[1:]:
            for node in layer:
                node.dOdZ = node.value * (1-node.value) #sigmoid specific
    
    def calculatedZdW(self):
        for layer in self.weightLayers:
            for weight in layer:
                weight.dZdW = weight.startNode.value
    
    def calculateHiddendEdO(self, layerIndex): #AKA dEdH
        for node in self.nodeLayers[layerIndex]:
            node.dEdO = sum([weight.endNode.dEdO * weight.endNode.dOdZ * weight.value for weight in node.outputWeights])
    
    def calculatedEdW(self):
        for layer in self.weightLayers:
            for weight in layer:
                weight.dEdW = weight.endNode.dEdO * weight.endNode.dOdZ * weight.startNode.value
    
    def calculatedEdB(self):
        for layer in self.nodeLayers[1:]:
            for node in layer:
                node.dEdB = node.dEdO * node.dOdZ * 1 #dZdB = 1
    
    def changeWeightsAndBiases(self):
        self.calculatedOdZ()
        self.calculatedZdW()
        for index in range(len(self.layerSizes)-2, 0, -1):
            self.calculateHiddendEdO(index)
        self.calculatedEdW()
        self.calculatedEdB()
        
        for layer in self.weightLayers:
            for weight in layer:
                weight.change -= self.learningRate*weight.dEdW
        for layer in self.nodeLayers[1:]:
            for node in layer:
                node.biasChange -= self.learningRate*node.dEdB
    
network = NeuralNetwork(networkShape, learningRate)
network.addTrainingData(trainingData)
while True:
    error = network.train()
    print("Error:", error)
