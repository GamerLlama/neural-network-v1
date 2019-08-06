#if order matters in the list of weights within a node, you're gonna have to redo some stuff
import random

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

class Weight:
    def __init__(self, startNode, endNode, startValue):
        self.startNode = startNode
        self.startNode.addOutput(self)
        self.endNode = endNode
        self.endNode.addInput(self)
        self.value = startValue
        self.ID = createNewID()
    
    def setValue(self, value):
        self.value = value

class Node:
    def __init__(self, layerType, startValue, startBias):
        self.layerType = layerType
        self.value = startValue
        self.bias = startBias
        self.inputWeights = []
        self.outputWeights = []
        self.ID = createNewID()
    
    def addInput(self, weight):
        self.inputWeights.append(weight)
        
    def addOutput(self, weight):
        self.outputWeights.append(weight)

class NeuralNetwork:
    def __init__(self, shape):
        self.layerSizes = shape
        self.createNodes()
        self.createWeights()

    def setInputs(self, newInputs):
        if len(newInputs)==self.layerSizes[0]:
            self.inputs = newInputs
        else:
            print("NEW INPUTS OF LENGTH", len(newInputs), "DO NOT MATCH INPUT NODE COUNT", self.layerSizes[0])

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
                self.nodeLayers[-1].append(Node(nodeType, 0, 0))
    
    def createWeights(self):
        self.weightLayers = []
        for index in range(len(self.layerSizes)-1):
            self.weightLayers.append([])
            for startNode in self.nodeLayers[index]:
                for endNode in self.nodeLayers[index+1]:
                    self.weightLayers[-1].append(Weight(startNode, endNode, 0))
        
networkShape = [10, 5, 5, 1]
network = NeuralNetwork(networkShape)