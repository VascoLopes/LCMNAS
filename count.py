from random import random
import torch
import torchvision.models as models
import torch.nn as nn
from torchcontrib.optim import SWA
from torchsummary import summary
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
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
import argparse


from modelsummary import torch_summary
from utils import Utils
from calc_tam_layer import *
from operations import * # operations (layers) in pytorch
from zero_cost_proxy import *
from processify import processify
from prunning import *
#import torch.multiprocessing as mp
#mp.set_start_method('spawn')


import time
import datetime
import sys
import traceback
import gc
from operator import itemgetter
#from memory_profiler import profile




#----------------------------------------------------------------------------#
''' ########################### '''

#----------------------------------------------------------------------------#

def generateDictStateTransitionGraph(modelSummary, printDictGraph=False):
    """Generate Markov Chain

    Arguments:
        modelSummary {[dict]} -- [Information about layer transition - Summary]

    Keyword Arguments:
        printDictGraph {bool} -- [If True, print markov chain at the end] (default: {False})

    Returns:
        [dict of dicts] -- [Markov Chain]
    """    
    
    print("#### Generating State Transition Graph [...] ####")

    # Count the frequencies
    stateTransitionFreq = {}
    predecessorLayer = "START"
    for index, row in modelSummary.iterrows():
        # Get layer name
        layerName = row['Layer'].split("-")[0]
        # If dict does not have last layer, add empty dict
        if predecessorLayer not in stateTransitionFreq.keys(): 
            stateTransitionFreq[predecessorLayer] = {}

        # Add layer transition
        if layerName not in stateTransitionFreq[predecessorLayer]:
            stateTransitionFreq[predecessorLayer][layerName] = 1
        else:
            stateTransitionFreq[predecessorLayer][layerName] = stateTransitionFreq[predecessorLayer][layerName]+1
        predecessorLayer = layerName

    # To add STOP node (previous->STOP)
    if predecessorLayer not in stateTransitionFreq.keys():
        stateTransitionFreq[predecessorLayer] = {}
    if 'END' not in stateTransitionFreq[predecessorLayer]:
        stateTransitionFreq[predecessorLayer]['END'] = 1
    else:
        stateTransitionFreq[predecessorLayer]['END'] = stateTransitionFreq[predecessorLayer]['END']+1

    if (printDictGraph):
        print("#-----------------------------#\n")
        print("---- Graph in the form of Dict ----")
        print (stateTransitionFreq)

    return (stateTransitionFreq)


def getLayerComponents(layerName):
    if ("Conv" in layerName):
        return ['out_channels', 'kernel_size']
    if (("AvgPool" in layerName) or ("MaxPool" in layerName)) and not ("Adaptive" in layerName):
        return ['kernel_size']
    if ("Linear" in layerName):
        return ['out_features']
    if ("Dropout" in layerName):
        return ['p']
    if ("Adaptive" in layerName):
        return ['output_size']
    if ("Bottleneck" in layerName):
        return ['out_channels']
    else:
        return ['None']


def generateHiddenMarkovStates(modelSummary, printHiddenStates=False):
    """Generate hidden states for each possible node in the markov chain, e.g.:
    conv2d -> (1*1:1,3*3:1,5*5:3,7*7:5)

    Returns:
        [dict of dict of dict] -- [Hidden Markov States that represent each layer and the frequency of components on it]
    """
    print("#### Generating Hidden Markov States [...] ####")

    hiddenMarkovStates = {}
    try:
        
        for index, row in modelSummary.iterrows():
            # Get layer name
            layerName = row['Layer'].split("-")[0]

            # If layer is not on the dict yet
            if layerName not in hiddenMarkovStates.keys(): 
                hiddenMarkovStates[layerName] = {}
            
            layerComponents = getLayerComponents(layerName)
            # Some layers don't require components (e.g. -> ReLU, ELU, ..)
            if layerComponents[0] == "None":
                #print (layerName)
                hiddenMarkovStates[layerName] = None
                continue
            for component in layerComponents: # Iterate over all possible components for given layer
                # Add component if not present
                if component not in hiddenMarkovStates[layerName]:
                    hiddenMarkovStates[layerName][component] = {}

                value = row[component] # Get value of the component
                # Check if value of the component is present in the dict, e.g: kernel_size: (3,3)
                #                                                               component   value
                if type(value) is list:
                    value = tuple(value)
                if str(value) not in hiddenMarkovStates[layerName][component].keys():
                    hiddenMarkovStates[layerName][component][value] = 1
                else:
                    hiddenMarkovStates[layerName][component][value] = hiddenMarkovStates[layerName][component][value]+1
    except Exception:
        print(traceback.format_exc())

    #print (type(list(hiddenMarkovStates['Conv2d']['kernel_size'].keys())[0]))
    return (hiddenMarkovStates)


#----------------------------------------------------------------------------#
def generateNetworkxDiGraph(stateTransitionFreq):
    '''
    Generate a Direct Graph in NetworkX from a ordereddict of orderedicts
    that contains the state transitions (and the number of occurences)
    '''
    ## Generate NetworkX graph and plot it
    G = nx.DiGraph(directed=True)
    G.edges.data('weight', default=1)
    # Add edges with weights
    for key in stateTransitionFreq:
        innerFrequencies = sum(stateTransitionFreq[key].values())
        #print(f'{key}  ->  {stateTransitionFreq[key]}   : {innerFrequencies}')
        G.add_node(key)
        for innerKey in stateTransitionFreq[key]:
            G.add_edge(key, innerKey, weight="%.2f"%(stateTransitionFreq[key][innerKey]/innerFrequencies))
            #print (f'i:{innerKey}, iv:{stateTransitionFreq[key][innerKey]}')
    return (G)


def generateGraphVizRepresentation(G, filename="filename.png"):
    print("\n#### Generating GraphViz Representation [...] ####")
    #nx.set_node_attributes(G, {k: {'label': labels[k]} for k in labels.keys()})
    nx.set_edge_attributes(G, {(e[0], e[1]): {'label': e[2]['weight']} for e in G.edges(data=True)})
    A = nx.drawing.nx_agraph.to_agraph(G)
    #print(A) # Print the graph structure
    A.layout('dot')
    A.draw(filename)

#----------------------------------------------------------------------------#
#-----------------------------Model Storing----------------------------------#
def writeDictPickle(modelDict, filename="./graphs/test.json"):
    print("\n#### Storing Dict as Pickle [...] ####")
    with open(filename, 'wb') as f:
        pickle.dump(modelDict, f)

def loadGraphDictPickle(filename="./graphs/test.json"):
    print("\n#### Reading Graph Dict Pickle [...] ####")
    with open(filename, 'rb') as f:
        loaded_dictionaries = pickle.load(f)
    return (loaded_dictionaries)

def writeGraphDict(modelDict, filename="./graphs/test.json"):
    print("\n#### Storing Dict as JSON [...] ####")
    with open(filename, 'w') as f:
        f.write(json.dumps(modelDict))

def loadGraphDict(filename="./graphs/test.json"):
    print("\n#### Reading Dict [...] ####")
    with open(filename, 'r') as read_file:
        loaded_dictionaries = json.loads(read_file.read())
    return (loaded_dictionaries)
#----------------------------------------------------------------------------#
#-----------------------------Create plots-----------------------------------#

def linePlot(xAxis, yAxis, labels, xLabel = "Epoch", yLabel ="Accuracy (%)", title=None, savePlot=False, fileName = "plot.pdf"):
    colors = ["blue", "red", "brown", "orange", "grey", "olive", "cyan", "purple", "pink"]
    plt.clf()

    if xAxis is not None:
        for index, x, y in enumerate(yAxis):
            plt.plot(xAxis, y, label=labels[index], color=colors[index])
    else:
        for index, y in enumerate(yAxis):
            plt.plot(y, label=labels[index], color=colors[index])

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    #plt.show()

    if savePlot:
        plt.savefig(fileName, bbox_inches='tight')


#----------------------------------------------------------------------------#
def readPopFromJSONs(path="./graphs/", args=None):
    torchModels = {}

    jsonFiles = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    for fileName in jsonFiles:
        modelName = fileName[:-5]
        torchModels[modelName] = OrderedDict()
        torchModels[modelName] = loadGraphDictPickle(path+fileName)

        if args.mixed_training: #combination
            torchModels[modelName]['fitness'] = args.mixed_fitness_lambda*torchModels[modelName]['score'] + (1-args.mixed_fitness_lambda)*torchModels[modelName]['valacc']
        elif args.without_training: # without train fitness
            torchModels[modelName]['fitness'] = torchModels[modelName]['score']
        else: # regular partial fitness #if args.without_training == False and args.mixed_fitness==False: 
            torchModels[modelName]['fitness'] = torchModels[modelName]['valacc']

    return (torchModels)


def generateSummaryAndGraph(model, img_shape = (3,244,244), automatedModel=True, input_image=0):
    try:
        modelSummary = torch_summary(img_shape, model, automatedModel, input_image)
        #print(modelSummary)
    except Exception as e:
        print (f'Exception generating model summary: {e}')
        return (None)
    
    # Because not every layer has the same columns (some may appear only once, others all the time)
    columns = set(['Layer'])
    # Add Column named - layer
    for key, value in modelSummary.items():
        colsAux = set(value.keys())
        columns = columns.union(colsAux)
    #print (columns)

    # Pass all values to a list in order to create a pandas df
    values = []
    for key,value in modelSummary.items():
        value['Layer'] = key
        values.append(value)
    df = pd.DataFrame(values, columns=columns, dtype=object) #[value for key,value in modelSummary.items()], columns = ms.keys()
    df = df.reindex(columns=(['Layer'] + list([a for a in df.columns if a != 'Layer']) )) #Layer should be the first column

    #print(df)
    #print (f'df: {type(df)}')

    stateTransitionFreq = generateDictStateTransitionGraph(df)
    hiddenMarkovStates  = generateHiddenMarkovStates(df)


    return stateTransitionFreq, hiddenMarkovStates # MarkovChain; Hidden States


def generateGraphsOfTorchVisionModels(utils, evaluateFitnes = True, data_dir = None, dataset = 'cifar10', datasetType='partial', 
        generateGraphVisualRepresentation=True, storeModelToFile=True, args=None, path="./models_pytorch/cifar10/partial/",
        trainloader=None, valloader=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get all pytorch models name
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    torchModels = {}
    for modelName in model_names:
        print(f"\n---- {modelName} [...] ----")
        print(modelName)

        # Get pytorch model

        try:
            #model = models.__dict__[modelName](pretrained=True).to(device)
            model = models.__dict__[modelName]().to(device)
        except:
            continue
        # Generate the stateTransition graph of the model and get the graph
        try:
            graph, hiddenStates = generateSummaryAndGraph(model, img_shape=(3,244,244), automatedModel=False)
            if graph is None: # some pytorch models can eventually produce errors
                continue
        except:
            print ("Model: "+modelName+", error creating Markov Chain.")
            continue

        torchModels[modelName] = OrderedDict()

        #stateTransitionFreq = generateDictStateTransitionGraph(modelSummary)
        torchModels[modelName]['graph'] = graph
        torchModels[modelName]['hiddenStates'] = hiddenStates
        

        # Get fitness of pytorch models
        fitness, valacc, score = 0, 0, 0
        # Get Criterion
        if (evaluateFitnes):
            fitness, valacc, score = getFitness(args, utils, model, trainloader, valloader, device, generated=False)
            '''
            criterion = utils.getCrossEntropyLoss()
            optimizer = utils.getSGDOptimizer(model)
            try:
                # Get dataloaders for the dataset
                trainloader, valloader, _ = utils.get_train_dataset(args, data_dir=data_dir, \
                                                            dataset=dataset, datasetType=datasetType, resize=True)

                print("#### Starting Training and Fitness Extraction [...] ####")
                for i in range(0,args.search_epochs): # Train for n epochs, evaluate
                    model, acc, _ = utils.trainNormalModels(model, trainloader, criterion, optimizer)
                    valacc, _ = utils.evaluateNormalModels(model, valloader, criterion)
            except Exception as e:
                print(e)
        #print (valacc)
        torchModels[modelName]['fitness'] = valacc
        print(f'Final Val Accuracy: {valacc}')


        # Get NAS without training score
        try:
            if args.without_training==True:
                model = models.__dict__[modelName]()
                model = utils.change_CNN_classifier(model, modelName, 1) #change for 1 class output
                model = model.to(device)
                bs = args.batch_size
                args.batch_size = 8 # change batch_size for score calculation
                # Get dataloader with required batch_size 
                trainloader, _, _ = utils.get_train_dataset(args, data_dir=data_dir, \
                                                        dataset=dataset, datasetType=datasetType, resize=True)
                torchModels[modelName]['score'] = zero_cost_proxy(trainloader, model, device, args.batch_size)
                print(f'Score: {torchModels[modelName]["score"]}')

                args.batch_size = bs # change batch_size for the original
        except Exception as e:
            print(e)
            pass
        '''
        torchModels[modelName]['valacc'] = valacc
        torchModels[modelName]['score'] = score
        torchModels[modelName]['fitness'] = fitness

        #torchModels[modelName]['model'] = model #copy.deepcopy(model.state_dict()) # Store the model
                                                # to rebuild it if needed

        if(generateGraphVisualRepresentation):
            modelGraph = generateNetworkxDiGraph(graph)
            generateGraphVizRepresentation(modelGraph, filename="./graphs/"+modelName+".png")
        
        if(storeModelToFile):
            writeDictPickle(torchModels[modelName], filename=path+modelName+".json")

        # The model itself is no longer needed
        del model
        #torch.cuda.empty_cache()
    return (torchModels)


def getFitness(args, utils, model, trainloader, valloader, device, generated=True):
    criterion = utils.getCrossEntropyLoss()
    optimizer = utils.getSGDOptimizer(model)

    fitness, val, score =0,0,0 #np.random.randint(100)
    # Acquire fitness
    try:
        # Acquire score that represents model fitness without requiring training
        if args.without_training or args.mixed_training:
            print("### Without Training [...] ###")
            score = zero_cost_proxy(trainloader, model, device, args.batch_size, desired_size=256, generated=generated)
    except Exception as e:
        score = 0
        print(e)
        if "out of memory" not in str(e):
            print(e)
            #print(modelLayersStringList)
            print(model)
        utils.clean_memory()
    try:
        # Train for some epochs to acquire validation accuracy of the generated model
        if args.mixed_training or (args.without_training == False and args.mixed_training == False):
            print("### Regular Training [...] ###")
            optimizer = utils.getSGDOptimizer(model)
            if generated: #the network is created using a modulelist
                for i in range(0, args.search_epochs): # Train for n epochs, evaluate
                    model, _, _, _ = utils.train(model, trainloader, criterion, optimizer)
                    top1, _, _ = utils.evaluate(model, valloader, criterion)
                    if (top1.avg > fitness):
                        val = top1.avg # store the best val accuracy (fitness)
            else: 
                for i in range(0, args.search_epochs): # Train for n epochs, evaluate
                    model, _, _ = utils.trainNormalModels(model, trainloader, criterion, optimizer)
                    valacc, _ = utils.evaluateNormalModels(model, valloader, criterion)
                    if (valacc > fitness):
                        val = valacc # store the best val accuracy (fitness)
    except Exception as e:
        val = 0
        print(e)
        if "out of memory" not in str(e):
            #print(modelLayersStringList)
            print(model)
        utils.clean_memory()

    if args.mixed_training: #combination
        fitness = args.mixed_fitness_lambda*score + (1-args.mixed_fitness_lambda)*val
    elif args.without_training: # without train fitness
        fitness = score
    else: # regular partial fitness #if args.without_training == False and args.mixed_fitness==False: 
        fitness = val

    return fitness, val, score


#----------------------------------------------------------------------------#
def checkIfLinearInProximity(list):
    for layer in list:
        if('Linear' in layer):
            return (True)
        if ("ReLU" in layer) or ("ELU" in layer) or ("SELU" in layer) or ("ReLU6" in layer) or ("Dropout" in layer):
            continue
        elif ("Conv2d" in layer) or ("AvgPool" in layer) or ("MaxPool" in layer) or ("BatchNorm2d" in layer):
            return (False)
    return (True) #is the last layer

def checkIfLinearInProximityReverse(list):
    list.reverse()
    for layer in list:
        if('Linear' in layer):
            return (True)
        if ("ReLU" in layer) or ("ELU" in layer) or ("SELU" in layer) or ("ReLU6" in layer) or ("Dropout" in layer):
            continue
        elif ("Conv2d" in layer) or ("AvgPool" in layer) or ("MaxPool" in layer) or ("BatchNorm2d" in layer):
            return (False)
    return (True) #is the last layer

def transformNetworkIntoPytorch(modelList, input_shape=[3,32,32], maxneurons=6000, n_classes=10):
    img_size = input_shape[1:]
    input_channels = input_shape[0]
    last_conv_output = input_shape[0]

    layers = nn.ModuleList()
    # Transform strings into pytorch layers
    #print (modelList)
    for idx, layerInformation in enumerate(modelList):
        stringLayer = layerInformation[0]
        components = layerInformation[1]

        if ("Conv" in stringLayer) and not ("Transpose" in stringLayer): #TODO: padding should be fixed using the calc_tam_layers, so the images does not get cropped
            out_channels = components['out_channels']
            kernel_size  = components['kernel_size']
            padding = Conv_Padding(kernel_size=kernel_size, img_sizes=img_size)
            layer = OPS[stringLayer](Cin = last_conv_output, Cout=out_channels, kernel_size=kernel_size, padding=padding)
            #layer = transformLayerNameIntoPytorch(stringLayer, in_channels=last_conv_output, out_channels=out_channels, kernel_size=kernel_size)
            last_conv_output = out_channels

        elif "Linear" in stringLayer:
            try:
                #if checkIfLinearInProximityReverse(modelList[:idx-1]):
                aux = modelList[:idx]
                aux.reverse()
                if checkIfLinearInProximity(aux):
                    in_features = out_features
                else:
                    in_features = Flatten(img_size = img_size, channels = last_conv_output)
            except:
                in_features = Flatten(img_size = img_size, channels = last_conv_output)
            keepFlatten = checkIfLinearInProximity(modelList[idx+1:])
            
            out_features = components['out_features'] if keepFlatten == True else in_features

            layer = OPS[stringLayer](in_features=in_features, out_features=out_features, unflatten=(not keepFlatten))

        elif "BatchNorm" in stringLayer:
            #layer = transformLayerNameIntoPytorch(stringLayer, num_features=last_conv_output)
            layer = OPS[stringLayer](last_conv_output)

        elif "Adaptive" in stringLayer:
            #print (components)
            out_features = components['output_size'] if components['output_size'] is not None else img_size
            #print (out_features)

            # if image size would be less than (3,..), 
            if (any(i < 3 for i in out_features)):
                out_features = img_size
            img_size = out_features
            
            layer = OPS[stringLayer](out_features=out_features)

        elif (("AvgPool" in stringLayer) or ("MaxPool" in stringLayer)) and not ("Adaptive" in stringLayer):
            #print ((components['kernel_size']))
            if isinstance(components['kernel_size'], int):
                kernel_size = components['kernel_size'] if all(i > components['kernel_size'] for i in img_size) else 1
                img_size = [int(elem/kernel_size) for elem in img_size] 
            else:
                kernel_size = components['kernel_size'] if all(i > c for c,i in zip(components['kernel_size'], img_size)) else 1
                # reduze img size accordingly
                img_size = [int(elem/k) for k,elem in zip(kernel_size, img_size)] 
            layer = OPS[stringLayer](kernel_size=kernel_size)                

        elif ("Dropout" in stringLayer): 
            dropout_prob = components['p']
            layer = OPS[stringLayer](p=dropout_prob)

        elif ("ReLU" in stringLayer) or ("ELU" in stringLayer) or ("SELU" in stringLayer):
            layer = OPS[stringLayer](False)

        elif ("Bottleneck" in stringLayer):
            out_channels = components['out_channels']
            stride = components['stride']
            # If the bottleneck does not downsample, out_channels must be equal to the input
            if stride == 1:
                out_channels, components['out_channels'] = int(last_conv_output/Bottleneck.expansion), int(last_conv_output/Bottleneck.expansion)
            layer = OPS[stringLayer](last_conv_output, out_channels, stride, None, 1, 64, 1, None)
            last_conv_output = layer.output
            img_size = [int(elem/stride) for elem in img_size] 

        else: # It will return None, for debug purposes
            print(stringLayer)
            layer = OPS[stringLayer]

        layers += [layer]

    iterator = 1
    while 1:
        layer = layers[-iterator]
        if isinstance(layer, LinearLayer):
            #print(f'img_size:{img_size}')
            in_features = layer.op.in_features
            #print(f'in_features:{in_features}')

            # Check if any of the last linear layers has too many in_features
            # That may cause overfitting
            if (in_features > maxneurons): # if too many neurons, try other model
                #print("oleeeeeeeeeee")
                return None
                
        elif not (isinstance(layer,nn.Dropout) or
            isinstance(layer,nn.ELU) or
            isinstance(layer,nn.ReLU) or
            isinstance(layer,nn.ReLU6) or
            isinstance(layer,nn.SELU)):
            break # there is no more MLP
        iterator = iterator+1

    # Adjust last linear layer for output_size = n_classes
    if isinstance(layers[-1], LinearLayer):
        if layers[-1].op.out_features != n_classes:
            layers[-1] = LinearLayer(in_features=layers[-1].op.in_features, out_features=n_classes)

    #model = nn.Sequential(*layers) #nn.ModuleList
    #print(layers)
    return (layers)

#----------------------------------------------------------------------------#
#-------------------------------Evolution------------------------------------#
def modelWeightedRouletteSelection(models, lastLayer):
    #Copy the dict, because dict parameters are references
    modelsDict = models.copy()
    ###print (f'Copied model: {modelsDict.keys()}')
    for k, v in list(modelsDict.items()):
        #print(f'key={k},  {v}')
        # Remove models that don't have HMC, the last layer or any fitness
        try:
            if ('graph' not in v) or not (lastLayer in v['graph']) or (v['fitness'] == 0):
                del modelsDict[k]
        except:
            del modelsDict[k]
    sortedPopulation = sorted(modelsDict.items(), key=lambda x: x[1]['fitness'])
    sortedPopulationNames = [i[0] for i in sortedPopulation] # Get only the names

    # Get rank by fitness of the previous population
    for key in modelsDict.keys():
        modelsDict[key]['rank'] = sortedPopulationNames.index(key)

    ###print (f'Copied model after del: {modelsDict.keys()}')
    max = sum(models[key]['rank'] for key in modelsDict) #'fitness'

    pick = np.random.uniform(0, max)
    current = 0
    for key in modelsDict:
        current += modelsDict[key]['rank'] #'fitness'
        if current > pick:
            return key

def layerSelectionFromModelGraph(graph, layer):
    max = sum(graph.values())
    pick = np.random.uniform(0, max)
    current = 0
    for key in graph:
        ###print(f'Key:{key}, value:{graph[key]}')
        current += graph[key]
        if current > pick:
            return key

def componentSelectionFromModelLayer(hiddenStates, component):
    """[summary]

    Arguments:
        hiddenStates {[dict]} -- [Has all the values of frequency for a given layer of the model]
        component {[String]} -- [Which component is to be used]
    """
    max = sum(hiddenStates[component].values())
    pick = np.random.uniform(0, max)
    current = 0
    for key in hiddenStates[component]:
        ###print(f'Key:{key}, value:{graph[key]}')
        current += hiddenStates[component][key]
        if current > pick:
            return key


def generateNewModel(models, img_size, n_classes=10, prob_residual = 50, prob_residual_downsample = 30):
    # Sample layers from start to end using roulette wheel selecetion
    # Based on models fitness
    lastLayer = "START"
    
    # Select parent model to generate a child new model
    #modelSelected = modelWeightedRouletteSelection(models, lastLayer)
    #print(modelSelected)

    newModelLayers=[]
    while(lastLayer != "END"):

        # Get model that will be encharged of the new layer
        # Select 1 model per layer
        modelSelected = modelWeightedRouletteSelection(models, lastLayer)
        if(modelSelected is None): # No model has that layer (defensive programming)
            break
        
        ###print(modelSelected)
        # All possible layers on the sampled model, from LastLayer: P(x+1|x), where x=LastLayer
        try:
            possibleChoises = models[modelSelected]['graph'][lastLayer]
        except:
            return None
        
        # Select the next layer
        if not args.random_search_layer:
            sampledLayer = layerSelectionFromModelGraph(possibleChoises, lastLayer)
        else:

            possibleChoises_keys = list(possibleChoises.keys())
            sampledLayer = possibleChoises_keys[np.random.randint(0,len(possibleChoises_keys))]

        # If sampled layer is the final one, end the generation
        # So that "END" layer does not get appended to the list
        if(sampledLayer=="END"):
            break

        # Get all possible components of a layer (kernel_size, output...)
        possibleComponents = getLayerComponents(sampledLayer)

        components = {}
        for component in possibleComponents:
            if ('None' in component):
                continue
            # Select component based on the probabilities
            if not args.random_search_layer:
                components[component] = componentSelectionFromModelLayer(models[modelSelected]['hiddenStates'][sampledLayer], component)
            else:
                possibleChoises = list(models[modelSelected]['hiddenStates'][sampledLayer][component])
                components[component] = possibleChoises[np.random.randint(0,len(possibleChoises))]

        if 'Conv' in sampledLayer:
            channels = components['out_channels']
        elif "Adaptive" in sampledLayer:
            img_size = components['output_size']
            #print(img_size)
        elif "Pool" in sampledLayer and "Adaptive" not in sampledLayer:
            if isinstance(components['kernel_size'], int):
                kernel_size = components['kernel_size'] if all(i > components['kernel_size'] for i in img_size) else 1
                img_size = [int(elem/kernel_size) for elem in img_size]
            else:
                kernel_size = components['kernel_size'] if all(i > c for c,i in zip(components['kernel_size'], img_size)) else 1
                # reduze img size accordingly
                img_size = [int(elem/k) for k,elem in zip(kernel_size, img_size)] 

        lastLayer = sampledLayer

        if 'Conv' in sampledLayer:
            # Probably, transform conv layer in residual connection
            pick = np.random.uniform(0, 100)
            if pick <= prob_residual:
                components['stride'] = 1
                pick = np.random.uniform(0, 100)
                # Probability of the bottleneck having downsampling
                if pick <= prob_residual_downsample:
                    # If the image size is already under 3 pixels in any dimension, does not downsample
                    if not (any(i < 3 for i in img_size)):
                        components['stride'] = 2
                        #print("OIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                        channels = components['out_channels'] * Bottleneck.expansion
                        
                # Calculate new img_size based on the stride of the bottleneck
                img_size = [int(Conv_Output(3, elem, components['stride'],1)) for elem in img_size]

                newModelLayers.append(('Bottleneck',components))
                lastLayer = 'ReLU'
            else:
                newModelLayers.append((sampledLayer,components))
        else:
            # Add sampled layer to the end of the model
            # Sampled layer becomes the x in P(x+1|x)
            newModelLayers.append((sampledLayer,components))


    #print(newModelLayers[-1])
    # Some generated networks might not end in linear layers (e.g, if squeezenet is the last parent)
    try:
        if 'Linear' not in newModelLayers[-1][0]:
            if 'ReLU' not in newModelLayers[-1][0]:
                newModelLayers.append(('ReLU',{}))
            if 'Dropout' not in newModelLayers[-1][0]:
                modelSelected = modelWeightedRouletteSelection(models, 'Dropout')
                components = {}
                if modelSelected is None:
                    components['p'] = 0.5 # Default for pytorch
                else:
                    components['p'] = componentSelectionFromModelLayer(models[modelSelected]['hiddenStates']['Dropout'], 'p')
                newModelLayers.append(('Dropout',components))
            newModelLayers.append(('Linear', {'out_features': n_classes}))
    except:
        return None
    
    # Adjust last linear layer for output_size = n_classes
    if 'Linear' in newModelLayers[-1][0] and newModelLayers[-1][1] != n_classes:
        #newModelLayers[-1] = ('Linear', {'out_features': n_classes}) # Original
        newModelLayers[-1] = ('Linear', {'out_features': n_classes}) 
    #print(newModelLayers[-1])

    return newModelLayers





def train_network(model, utils, args, train_loader, val_loader):
    # Get Criterion
    criterion = utils.getCrossEntropyLoss()
    # Train the model for n epochs (normal training)
    # Store model in best val acc, best val loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_lowest_loss = []#copy.deepcopy(model.state_dict())
    best_acc_val = [0.0, 0] # accuracy, epoch
    best_loss_val = [20.0, 0] # loss epoch
    # Store each epoch evolution
    all_train_acc  = []
    all_train_loss = []
    all_val_acc    = []
    all_val_loss   = []

    # Create SGD Optimizer for this model
    optimizer = utils.getSGDOptimizer(model, learningRate=args.learning_rate,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Implemented SWA
    #optimizer = SWA(optimizer, swa_start=50, swa_freq=10, swa_lr=learning_rate)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)#, eta_min=0.001)

    startTimeFinalTrain = time.time() # Store time that it takes to train final model
    print("### Start Regular Training ###")
    for epoch in range(0,args.epochs):
        print ("-----")
        
        # Train
        model, top1, top5, train_loss = utils.train(model, train_loader, criterion, optimizer, epoch)
        all_train_acc.append(top1.avg)
        all_train_loss.append(train_loss.avg)
        
        # Learning Rate Decay Scheduler
        scheduler.step()

        # Validation
        val_top1, val_top5, val_loss = utils.evaluate(model, val_loader, criterion, epoch)
        all_val_acc.append(val_top1.avg)
        all_val_loss.append(val_loss.avg)

        if val_top1.avg > best_acc_val[0]: # store best model so far, for later, based on best val acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc_val[0], best_acc_val[1] = val_top1.avg, epoch
        #if val_loss.avg < best_loss_val[0]: # store best model according to loss
        #    best_model_lowest_loss = copy.deepcopy(model.state_dict())
        #    best_loss_val[0], best_loss_val[1] = val_loss.avg, epoch

        if args.prunning:
            model = unstructured_prune(model, threshold=args.prunning_threshold)

        #print (f'Epoch {epoch:3} -> Train Accuracy: {top1.avg}, Loss: {train_loss},   Val Accuracy: {val_acc}')
        print('Epoch {epoch:3} : ' 
            'Acc@1: {top1.avg:.3f}\t\t'
            'Acc@5: {top5.avg:.3f}\t\t'
            'Val Acc@1: {val_top1.avg:.4f}\t\t' 
            'Val Acc@5: {val_top5.avg:.4f}\t\t' 
            'Train Loss: {train_loss.avg:.4f}\t\t' 
            'Val Loss: {val_loss.avg:.4f}\t\t' 
            .format(
                epoch=epoch, top1=top1, top5=top5, 
                val_top1=val_top1, val_top5=val_top5, 
                train_loss=train_loss, val_loss=val_loss))

    endTimeFinalTrain = time.time() - startTimeFinalTrain # Store time that it takes to train final model

    #optimizer.swap_swa_sgd()
    
    return model, best_model_wts, best_model_lowest_loss, all_train_acc, all_train_loss, all_val_acc, all_val_loss, best_acc_val, best_loss_val, endTimeFinalTrain


#----------------------------------------------------------------------------#
#----------------------------------MAIN--------------------------------------#
def main(args, save_to = "./experiments/"):
    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir, batch_size, generations, populationSize, elitism, probResidual, probResidualDownsample, epochs, \
        learning_rate, weight_decay, maxneurons, epochsTrainSearch, dataset, datasetType, cutout, cutout_length, \
            auto_augment \
            = \
            args.data_dir, args.batch_size, args.generations, args.population, args.elitism, args.prob_residual, \
            args.prob_residual_downsample, args.epochs, args.learning_rate, args.weight_decay, args.max_neurons, \
            args.search_epochs, args.dataset.lower(), args.dataset_type.lower(), args.cutout, args.cutout_length, \
            args.auto_augment

    initPopDefined = args.init_pop
    modelsPath     = './models_pytorch/'
    if not os.path.exists(modelsPath):
            os.makedirs(modelsPath)
    modelsPath = os.path.join(modelsPath, dataset+'/')
    modelsPath = os.path.join(modelsPath, datasetType+'/') if "Fly" not in datasetType.lower() else os.path.join(modelsPath, 'partial/')

    # Store time for everything
    startTime      = time.time()

    # Create utils object (for train, optimizers, ...)
    utils = Utils(batch_size, device)

    # Get initial population
    # generate dictionary of models - initial population based on human-designed models
    if initPopDefined == True:
        # Read population from json
        population = readPopFromJSONs(path=modelsPath, args=args)
        if args.random_search_model:
            for key, _ in population.items():
                population[key]['fitness'] = 10
    else: 
        trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=data_dir, \
                                                        dataset=dataset, datasetType=datasetType, resize=True)
        # Generate search space using pytorch models
        startTimeGenerateSearchSpace = time.time()
        
        if not os.path.exists(modelsPath):
            os.makedirs(modelsPath)
        population = generateGraphsOfTorchVisionModels(utils, True, data_dir, dataset, datasetType, True, 
                storeModelToFile=True, args=args, path=modelsPath, trainloader=trainloader, valloader=valloader)
        endTimeGenerateSearchSpace = time.time()
        
        print(f'Time taken to generate complete spearch space: {endTimeGenerateSearchSpace-startTimeGenerateSearchSpace}')
        exit()
        


    # Get dataloaders for the dataset
    ##trainloader, valloader, n_classes = data_loader_cifar10.get_train_valid_loader("./data/", batch_size,
    ##            datasetType=datasetType)
    trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=data_dir, \
                                                            dataset=dataset, datasetType=datasetType)
    #testloader = utils.get_test_dataset(args, data_dir="./data/", dataset=dataset)

    # Get image shape for this dataset/problem
    for train_images, _ in trainloader:  
        sample_image = train_images[0]
        img_size = sample_image.shape # image size of the given problem
        break    
    print(f'Image size: {img_size}')


    # Get Criterion
    criterion = utils.getCrossEntropyLoss()

    startTimeSearch = time.time() # Store time that takes to perform search
    generationInfos = {} # store information about each generation
    for gen in range(0,generations):
        #trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=data_dir, \
        #                                                dataset=dataset, datasetType=datasetType)
        print(f"### Starting Generation {gen} [...] ###")

        # newPopulation will hold the new individuals (generated models)
        newPopulation = {}

        # Elitism : the best model from last gen continues
        #sortedPopulation -> tuples: (name, MODELINFO)
        sortedPopulation = sorted(population.items(), key=lambda x: x[1]['fitness'], reverse=True)
        del population
        utils.clean_memory()

        # avoid cases where the initial generation (search space) is much lower than the populationsize
        elitism = args.elitism if args.elitism < len(sortedPopulation) else len(sortedPopulation)
        
        # Elitism for generation 0, only 'search space models' are available
        if gen == 0:
            for i in range(elitism):
                newPopulation[i] = OrderedDict()
                newPopulation[i]['graph'] = sortedPopulation[i][1]['graph']
                newPopulation[i]['hiddenStates'] = sortedPopulation[i][1]['hiddenStates']
                newPopulation[i]['fitness'] = sortedPopulation[i][1]['fitness']
        # Elitism for the rest of the generations
        else:
            iterator, countElitism, flagInitialModels = 0, 0, 1 #flaginitialmodels to 1, does not allow any of the search space models to be passed to next gens
            while(True):
                if countElitism == elitism:
                    break
                #print(sortedPopulation)
                if 'model' not in sortedPopulation[iterator][1] and flagInitialModels == 1:
                    iterator = iterator + 1
                    continue

                newPopulation[countElitism] = OrderedDict()
                newPopulation[countElitism]['graph']        = sortedPopulation[iterator][1]['graph']
                newPopulation[countElitism]['hiddenStates'] = sortedPopulation[iterator][1]['hiddenStates']
                newPopulation[countElitism]['fitness']      = sortedPopulation[iterator][1]['fitness']
                if 'model' in sortedPopulation[iterator][1]: # initial pytorch models are ditched for memory optimization
                    newPopulation[countElitism]['model']    = sortedPopulation[iterator][1]['model']
                else:
                    flagInitialModels = 1

                iterator = iterator + 1
                countElitism = countElitism + 1

        print(f'Dictionary sizes ->  newPopulation:{utils.get_size(newPopulation)} \t sortedPopulation:{utils.get_size(sortedPopulation)}')

        # transform list of Tuple(int, orderedict) into ordereddict for easier management
        sortedPopulation = OrderedDict(sortedPopulation)

        # Generate the rest of the population
        for individual in range(elitism,populationSize): #[elitism,popsize[ because the inicial indexes are the best from last generation
            # Generate individual based on the graphs from last generation
            modelLayersStringList, model = None, None # Initialize model variables
            graph, hiddenStates = None, None # Initilize HMC variables
            print("Generating new model [...]")
            while modelLayersStringList is None or model is None: # Generate models until one is ok
                # Generate a new model
                try:
                    modelLayersStringList = generateNewModel(sortedPopulation, img_size[1:], n_classes=n_classes, prob_residual = probResidual, \
                                                            prob_residual_downsample = probResidualDownsample)
                except:
                    #traceback.print_exc()
                    modelLayersStringList = None
                    utils.clean_memory()

                if modelLayersStringList is None:
                    continue

                lengthModel = 0
                for layer, components in modelLayersStringList:
                    if 'Bottleneck' in layer:
                        if components['stride'] == 2:
                            lengthModel += 8
                        else:
                            lengthModel += 7
                    else:
                        lengthModel += 1
                if lengthModel > 500: # Trying to avoid RAM segmentation fault
                    modelLayersStringList = None
                    continue

                # Transform model into pytorch
                try:
                    num_classes=n_classes
                    if args.without_training and args.mixed_training == False:
                        num_classes = 1
                    model = processify(transformNetworkIntoPytorch, modelLayersStringList, input_shape=img_size, 
                                                        maxneurons=maxneurons, n_classes=num_classes)
                    #model = transformNetworkIntoPytorch(modelLayersStringList, input_shape=img_size, 
                    #                                    maxneurons=maxneurons, nClasses=num_classes)
                    if model is None:
                        continue
                    model = model.to(device)
                except Exception as e:
                    #traceback.print_exc()
                    print(e)
                    model = None #too big for memory
                    #exit()
                    utils.clean_memory()
                    if 'broken pipe' in str(e.__str__).lower():
                        torch.cuda.empty_cache()
                    continue

                # Generate Hidden-Markov Chain
                try:
                    graph, hiddenStates = generateSummaryAndGraph(model, img_shape=img_size, 
                                                        automatedModel=True, input_image=sample_image)
                except Exception as e:
                    #print (model)
                    #print(modelLayersStringList)
                    #if "NoneType" in str(e):
                    print(f'Individual {individual} not capable of generating Hidden Markov Chain')
                    print (e)
                    del model
                    model = None
                    utils.clean_memory()
                    continue

            #print (modelLayersStringList)

            # Add individual to the pool
            newPopulation[individual] = OrderedDict()

            fitness = 0 #np.random.randint(100)

            # Acquire fitness
            print("#### Starting Fitness Extraction [...] ####")
            try:
                if args.random_search_model:
                    fitness = 10 #random, equal to all models
                else: 
                    fitness, _,_ = getFitness(args, utils, model, trainloader, valloader, device)
            except:
                fitness = 0
            #print (fitness)
            '''
            try:
                # Acquire score that represents model fitness without requiring training
                if args.without_training or args.mixed_training:
                    print("### Without Training [...] ###")
                    score = zero_cost_proxy(trainloader, model, device, batch_size, desired_size=256, generated=True)
            except Exception as e:
                score = 0
                print(e)
                if "out of memory" not in str(e):
                    print(modelLayersStringList)
                    print(model)
                utils.clean_memory()
            try:
                # Train for some epochs to acquire validation accuracy of the generated model
                if args.without_training == False or args.mixed_training:
                    print("### Regular Training [...] ###")
                    optimizer = utils.getSGDOptimizer(model)
                    for i in range(0, args.search_epochs): # Train for n epochs, evaluate
                        model, _, _, _ = utils.train(model, trainloader, criterion, optimizer)
                        top1, _, _ = utils.evaluate(model, valloader, criterion)
                        if (top1.avg > fitness):
                            val = top1.avg # store the best val accuracy (fitness)
            except Exception as e:
                val = 0
                print(e)
                if "out of memory" not in str(e):
                    print(modelLayersStringList)
                    print(model)
                utils.clean_memory()
            


            if args.mixed_training: #combination
                fitness = args.mixed_fitness_lambda*score + (1-args.mixed_fitness_lambda)*val
            elif args.without_training: # without train fitness
                fitness = score
            else: # regular partial fitness #if args.without_training == False and args.mixed_fitness==False: 
                fitness = val
            '''
            #print(f'Fitness:{fitness}')
            # Fitness is the accuracy in validation with 1 epoch train or score from NASwithout training
            newPopulation[individual]['fitness'] = fitness 

            # Add the hidden markov chain
            newPopulation[individual]['graph'] = graph
            newPopulation[individual]['hiddenStates'] = hiddenStates

            newPopulation[individual]['model'] = modelLayersStringList
            print("\n")


        #print(f'Individual:{individual}, graph:{graph}, fitness:{valacc}')
        individuals, sumFitnesses = 0, 0
        print(f'Gen:{gen} Stats:')
        for k,v in newPopulation.items():
            individuals += 1
            sumFitnesses +=v["fitness"]
            print(f'Individual:{k} - fitness:{v["fitness"]}')
        print(f'Mean Fitness:{sumFitnesses/individuals:.3f}')
        print("-- End Generation --\n")
        
        
        #print(h.heap())

        # to store the information of the best model each epoch
        sortedPopulation = sorted(newPopulation.items(), key=lambda x: x[1]['fitness'], reverse=True)
        generationInfos[gen] = {}
        generationInfos[gen]['numberIndvs'] = individuals
        generationInfos[gen]['meanFitness'] = sumFitnesses/individuals  
        generationInfos[gen]['bestModelFitness'] = sortedPopulation[0][1]['fitness']
        #generationInfos[gen]['graph'] = sortedPopulation[0][1]['graph']
        #if 'model' in sortedPopulation[0][1]: # initial pytorch models are ditched for memory optimization
        #    generationInfos[gen]['model'] = sortedPopulation[0][1]['model']

        population = copy.deepcopy(newPopulation)

        try:
            # Free memory
            del model
            del modelLayersStringList
            del newPopulation
            # Free Torch memory & Call Garbage collector
            utils.clean_memory()
        except:
            pass

    endTimeSearch = time.time() - startTimeSearch # Store final time after search



    print("#### Starting Best Model Selection [...] ####")
    trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=data_dir, \
                                                        dataset=dataset, datasetType=datasetType)#datasetType)
    # Get best NAS generated model
    sortedPopulation = sorted(population.items(), key=lambda x: x[1]['fitness'], reverse=True)
    if args.without_training or (args.mixed_training and args.search_epochs < args.best_model_search_epochs) or (not args.without_training and not args.mixed_training and args.search_epochs < args.best_model_search_epochs):
        newPopulation = OrderedDict()
        iterations = len(sortedPopulation) if  len(sortedPopulation) < args.final_indv_without_train else args.final_indv_without_train
        for i in range(0, iterations): # default 10
            network = sortedPopulation[i][1]
            try: 
                model = transformNetworkIntoPytorch(network['model'], input_shape=img_size, maxneurons=maxneurons, n_classes=n_classes)
                model = model.to(device)
            except Exception as e: # A model from search space might happen to get till the end over the elitism, remove such choice (Defense Programming)
                print(e)
                args.final_indv_without_train = args.final_indv_without_train+1
                continue
            
            optimizer = utils.getSGDOptimizer(model, args.learning_rate)
            print("#### Starting Fitness Extraction [...] ####")
            fitness = 0#network['fitness']
            try:
                for _ in range(0, args.best_model_search_epochs): # Train for n epochs, evaluate
                    model, _, _, _ = utils.train(model, trainloader, criterion, optimizer)
                    top1, _, _ = utils.evaluate(model, valloader, criterion)
                    print(top1.avg)
                    if (top1.avg > fitness):
                        fitness = top1.avg # store the best val accuracy (fitness)
            except Exception as e: #out of memory model
                print(traceback.format_exc())
                print(e)
                pass

            del model
            utils.clean_memory()

            newPopulation[i] = OrderedDict()
            newPopulation[i]['graph'] = network['graph']
            newPopulation[i]['hiddenStates'] = network['hiddenStates']
            newPopulation[i]['model'] = network['model']
            newPopulation[i]['fitness'] = fitness

        sortedPopulation = (sorted(newPopulation.items(), key=lambda x: x[1]['fitness'], reverse=True))

        print (f'Top {iterations} models after {epochsTrainSearch} epochs trained [...]')
        for k,v in sortedPopulation:
            print(f'Individual:{k} - fitness:{v["fitness"]}')


    # Select the best model!
    for model in sortedPopulation:
        bestModel = model[1]
        # If the best model is from the search space, continue until a generated model
        if 'model' in bestModel:
            break


    print ("\n### Final Model Training [...] ###")
    # Train model
    bestval_acc_each=[]
    for i in range(1): #(3):
        if i == 1:
            #args.auto_augment = True
            args.prunning = True 
            
        # Get final dataset
        # Final train, val,test loader
        try:
            trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=data_dir, \
                                                                    dataset=dataset, datasetType='TrainEntire', shuffle=True, \
                                                                    auto_augment=args.auto_augment)
            print(n_classes)

            model = bestModel['model']
            #if i == 0:
            #    model = utils.add_bn(model)
            #elif i==1:
            #    model = utils.add_bn(model, False)

            # Transform most performant NAS model to pytorch network
            #model = transformNetworkIntoPytorch(bestModel['model'], input_shape=img_size, maxneurons=maxneurons).to(device)
            model = transformNetworkIntoPytorch(model, input_shape=img_size, maxneurons=maxneurons, n_classes=n_classes)
            print(model)
            model = model.to(device)

            model, best_model_wts, best_model_lowest_loss, all_train_acc, all_train_loss, all_val_acc, \
                all_val_loss, best_acc_val, best_loss_val, endTimeFinalTrain \
                = train_network(model, utils, args, trainloader, valloader)

            bestval_acc_each.append(best_acc_val[0])
        except Exception as e:
            print (e)
            continue
    print(model)

    utils.clean_memory()

    model.load_state_dict(best_model_wts)

    # Test model on new data
    testloader = utils.get_test_dataset(args, data_dir=data_dir, dataset=dataset)
    accuracy_test, inference_time = utils.test(model, testloader)

    print(f'trainAcc={all_train_acc}')
    print(f'trainLoss={all_train_loss}')
    print(f'valAcc={all_val_acc}')
    print(f'valLoss={all_val_loss}')
    print(f'Final Test Accuracy: {accuracy_test} : val accuracy {best_acc_val[0]} - train accuracy {all_train_acc[best_acc_val[1]]}')
    
    #model.load_state_dict(best_model_lowest_loss)
    #accuracy_test, inference_time = utils.test(model, testloader)
    #accuracy_test = all_val_acc[best_loss_val[1]]
    #print(f'Final Test Accuracy WITH LOWEST LOSS: {accuracy_test} : val accuracy {all_val_acc[best_loss_val[1]]} - train accuracy {all_train_acc[best_loss_val[1]]}')

    # Print best individual for each generation and mean fitness
    for k,v in generationInfos.items():
        print (f'Generation:{k} -> Best Individual:{generationInfos[k]["bestModelFitness"]}; Mean Fitness:{generationInfos[k]["meanFitness"]}')

    # Times
    print(f'Time taken to perform search: {endTimeSearch}')
    print(f'Time taken to train the final model from scratch: {endTimeFinalTrain}')
    print(f'Time taken for the whole process:{time.time()-startTime}')

    try:
        model.eval()
        print(f'Model parameters:{utils.count_parameters(model)}')
        print(f'Inference Time Mean: {np.mean(inference_time)}, STD:{np.std(inference_time)} ms')
    except:
        pass

    '''
    print(f'Validation accuracy of the model using different Autoaugment schemes:')
    print(f'Normal model train: {bestval_acc_each[0]}')
    print(f'Auto Augment: {bestval_acc_each[1]}')
    '''

    '''
    print(f'Validation accuracy of the model using different BN schemes:')
    print(f'Normal model : {bestval_acc_each[2]}')
    print(f'All layers BN: {bestval_acc_each[0]}')
    print(f'non-activ  BN: {bestval_acc_each[1]}')
    '''

    torch.save(model, save_to+"model.pth")


    modelGraph = generateNetworkxDiGraph(bestModel['graph'])
    year, month, day, hour, min = map(int, time.strftime("%Y %m %d %H %M").split())
    generateGraphVizRepresentation(modelGraph, filename=save_to+"graphviz.png")
    #writeGraphDict(stateTransitionFreq, filename="./graphs/experiments.json")

    # Generate Plots
    ## Best Indv per generation
    fitnesses = [d['bestModelFitness'] for d in generationInfos.values()]
    linePlot(None, [fitnesses], ["Best Individual"], xLabel="Generation", yLabel="Fitness (%)", title=None, savePlot=True, 
        fileName=save_to+".bestindv.pdf")
    ## Mean Fitness per generation    
    fitnesses = [d['meanFitness'] for d in generationInfos.values()]
    linePlot(None, [fitnesses], ["Mean Fitness"], xLabel="Generation", yLabel="Fitness (%)", title=None, savePlot=True, 
        fileName=save_to+".meanfitness.pdf")
    ## Final Model Train Accuracy
    linePlot(None, [all_train_acc, all_val_acc], ["Train", "Validation"], xLabel="Epoch", yLabel="Accuracy (%)", title=None, savePlot=True, 
        fileName=save_to+".accuracy.pdf")
    ## Final Model Train Loss
    linePlot(None, [all_train_loss, all_val_loss], ["Train", "Validation"], xLabel="Epoch", yLabel="Loss", title=None, savePlot=True, 
        fileName=save_to+".loss.pdf") #str(year)+"-"+str(month)+"-"+str(day)+":"+str(hour)+"_"+str(datasetType)+



if __name__ == "__main__":
    parser = argparse.ArgumentParser("UMNAS")
    parser.add_argument('--data_dir', type=str, default='./data/', help='location of the datasets')
    parser.add_argument('--output_path', type=str, default='./experiments/', help='directory to where all experiments will be dumped')
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset folder')
    parser.add_argument('--dataset_type', type=str, default='partial', help='Type of the dataset (ex: Full, Partial, ..)')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    parser.add_argument('--search_epochs', type=int, default=4, help='num of epochs to train searched models')
    parser.add_argument('--best_model_search_epochs', type=int, default=10, help='num of epochs to select the best model searched')
    parser.add_argument('--without_training', action='store_true', default=False, help='use NAS without training for search') 
    parser.add_argument('--mixed_training', action='store_true', default=False, help='acquire fitness with partial train +without training') 
    parser.add_argument('--final_indv_without_train', type=int, default=10, help='num indvs to train at the end of the search, using without train')
    parser.add_argument('--mixed_fitness_lambda', type=float, default=0.75, help='percentage of importance given to without train (1-lambda for val)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--population', type=int, default=25, help='population size')
    parser.add_argument('--generations', type=int, default=50, help='number of generations')
    parser.add_argument('--max_neurons', type=int, default=6000, help="number of max input neurons in final MLP layers")
    parser.add_argument('--prob_residual', type=int, default=50, help='probability of conv becomming residual block')
    parser.add_argument('--prob_residual_downsample', type=int, default=30, help='probability of residual block includes downsampling')
    
    parser.add_argument('--init_pop', action='store_false', default=True, help='init population defined') 
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout') 
    parser.add_argument('--auto_augment', action='store_true', default=False, help='use auto augment')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--random_search_layer', action='store_true', default=False, help='apply random search on layer selection (baseline)')
    parser.add_argument('--random_search_model', action='store_true', default=False, help='apply random search on model selection (baseline)')

    parser.add_argument('--prunning', action='store_false', default=True, help='use prune in final training') #TODO true?
    parser.add_argument('--prunning_threshold', type=float, default=1e-2, help='threshold value for prunning (abs(x)>threshold)') #TODO true?
    

    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')


    parser.add_argument('--elitism', type=int, default=int((0.15*parser.parse_args().population) if int(0.15*parser.parse_args().population > 0) else 1), help="number of elements to pass through elitism")


    args = parser.parse_args()

    '''
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    '''
    
    # check if experiments folder already exists
    experiments_path="./experiments/"
    if not os.path.exists(experiments_path):
        os.makedirs(experiments_path)
    d = datetime.datetime.now()
    iterator = 0
    while (1):
        path = os.path.join(experiments_path, "exp"+str(iterator)+"_"+d.strftime("%d_%b")+"/")
        print(path)
        if (os.path.exists(path)):
            iterator += 1
            continue
        os.makedirs(path)
        break
    
    print(args)

    main(args, save_to = path)
    sys.exit()