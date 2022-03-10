import torch
import torchvision.models as models
import torch.nn as nn
import pretrainedmodels
from torchsummary import summary
from torch.autograd import Variable
from collections import OrderedDict
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv
import json
import copy
import numpy as np
import pickle
import os

from modelsummary import summaryX
from utils import Utils
from calc_tam_layer import *
import data_loader_cifar10
from operations import * # Pytorch operations (layers)
import time

from simplecnn import SimpleCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generateSummary(model, img_shape = (3,244,244), automatedModel=True, input_image=0):
    try:
        modelSummary = summaryX(img_shape, model, automatedModel, input_image)
    except Exception as e:
        print (f'Exception generating model summary: {e}')
        return (None)
    return (modelSummary)

def generateModels():
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    torchModels = {}
    for modelName in model_names:
        print(f"\n---- {modelName} [...] ----")
        print(modelName)

        model = models.__dict__[modelName]().to(device)
        print("[ Generating Summary ... ]")
        # Generate the stateTransition graph of the model and get the graph
        graph = generateSummary(model, img_shape=(3,244,244), automatedModel=False)
        
        print("[ Storing Summary ... ]")
        if graph is not None:
            torchModels[modelName] = OrderedDict()

            #stateTransitionFreq = generateDictStateTransitionGraph(modelSummary)
            torchModels[modelName]['summary'] = graph
        else:
            continue

        # The model itself is no longer needed
        del model
        torch.cuda.empty_cache()
        
    return torchModels


def getParamDependingLayer(layerName):
    layerName = layerName.lower()
    if "conv" in layerName or ("pool" in layerName and "adaptive" not in layerName):
        return "kernel_size"
    elif "adaptive" in layerName:
        return "output_size"
    elif "dropout" in layerName:
        return "p"
    elif "linear" in layerName:
        return "out_features"
    else:
        return None


#Count operations
if __name__ == "__main__":
    population = generateModels()

    operations = {}
    for ind in population.keys():
        print (population[ind]['summary'].keys())
        for layerName, values in population[ind]['summary'].items():
            layerName = layerName.split("-")[0]
            #print(values)
            param = getParamDependingLayer(layerName)
            if layerName not in operations.keys():
                operations[layerName] = []
                if param is not None:
                    operations[layerName] += [values[param]]
                #else:
                #    print(f"Layer: {layerName}, had {param}")
            else:
                if param is not None:
                    operations[layerName] += [values[param]]
                #else:
                #    print(f"Layer: {layerName}, had {param}")
                #operations[]
            #exit()
    
    #  Remove duplicates
    for layer in operations:
        operations[layer] = set(operations[layer])

    # count operations
    count = 0
    for layer in operations:
        size = len(operations[layer])
        if size == 0:
            count += 1 #relu, etc
        else:
            count += size
    print (operations)
    print(f'{count} operations!')
