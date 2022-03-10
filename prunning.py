import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import prune
from operations import LinearLayer

class ThresholdPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured" #global, structured, unstructured

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold


def prunable_layers(model):
    parameters_to_prune = [
    (child, "weight")
    for child in model.children()
    if hasattr(child, 'weight')#isinstance(child, torch.nn.Conv2d) 
    ]
    return parameters_to_prune

def unstructured_prune(model, threshold=1e-2):
    parameters_to_prune = prunable_layers(model)
    prune.global_unstructured(
        parameters_to_prune, pruning_method=ThresholdPruning, threshold=threshold
    )
    
    for child in model.children():
        if hasattr(child, 'weight'):
            prune.remove(child, "weight")
    return model

def test():
    network = nn.ModuleList()
    network.append(nn.Conv2d(3,64, 3))
    network.append(nn.Conv2d(64,64, 3))
    network.append(nn.Conv2d(64,256, 3))
    network.append(nn.Conv2d(256,256, 3))
    #network.append(nn.AvgPool2d(3))
    network.append(nn.AdaptiveAvgPool2d((4,4)))
    network.append(LinearLayer(4*4*256,10, False))
    print(network)
    input = torch.randn(1,3,32,32)
    output = input
    for i in network:
        output = i(output)

    # Check initial weights
    for child in network.children():
        if hasattr(child, 'weight'):
            print (child.weight)
        break

    # Globally and permanently prune, layer by layer using a threshold (absolute value)
    network = unstructured_prune(network, threshold = 1e-2)

    # Verify new weights
    for child in network.children():
        if hasattr(child, 'weight'):
            print (child.weight)
        break

if __name__ == "__main__":
    test()
