import torch
import torchvision.models as models
import torch.nn as nn
import pretrainedmodels
from torch.autograd import Variable
from collections import OrderedDict
from processify import processify

def torch_summary(input_size, model, automatedModel=False, input_image=0):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            #print(module_idx)
            
            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            #print(outp)
            #print("ole")
            try:
                summary[m_key]['input_shape'] = list(input[0].size())
            except:
                summary[m_key]['input_shape'] = [-1]
            #print("ole2")
            
            #summary[m_key]['input_shape'][0] = -1
            try:
                summary[m_key]['output_shape'] = list(output.size())
            except:
                summary[m_key]['output_shape'] = [-1]
            #print("ole3")
            #print("-----------------------")
            
            #summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

            if hasattr(module, 'kernel_size'): # Get kernel_size
                #print (class_name)
                summary[m_key]['kernel_size'] = module.kernel_size
                #print ((module.kernel_size))
            if hasattr(module, 'num_features'):
                #print (module.num_features)
                summary[m_key]['num_features'] = module.num_features
            if hasattr(module, 'out_channels'): # Conv layers,..
                summary[m_key]['out_channels'] = module.out_channels
            
            if hasattr(module, 'in_features'): # Linear layers
                summary[m_key]['in_features'] = module.in_features
            if hasattr(module, 'out_features'):
                summary[m_key]['out_features'] = module.out_features

            if hasattr(module, 'input_size'):
                summary[m_key]['input_size'] = module.input_size
            if hasattr(module, 'output_size'):
                summary[m_key]['output_size'] = module.output_size
            if hasattr(module, 'p'): #probability -> dropout..
                summary[m_key]['p'] = module.p

            if hasattr(module, 'hidden_size'): #LSTMS, RNNS...
                summary[m_key]['hidden_size'] = module.hidden_size


        className = str(module.__class__).split('.')[-1].split("'")[0]

        #Don't append the name of blocks, only the layers
        # Many ifs are due to pytorch models' classes
        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model) and \
                not ("Bottleneck" in className) and \
                not ("InceptionAux" in className) and \
                not ("Inception" in className) and \
                not ("InvertedResidual" in className) and \
                not ("BasicBlock" in className) and \
                not ("Fire" in className) and \
                not ("BasicConv2d" in className) and \
                not ("DenseBlock" in className) and \
                not ("LinearLayer" in className) and \
                not ("DenseLayer" in className):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network

    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size))


    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    x = x.cuda() if torch.cuda.is_available() else x

    if not automatedModel:
        model(x) 
    else:
        for layer in model:
            #print(x.shape)
            x = layer(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    return summary


# Auxiliar different print types
'''
# Normal model print
print(3*"#-----------------------------#\n")
print(model)
'''

'''
print(3*"#-----------------------------#\n")
for name, param in model.named_parameters():

    if "weight" in name:
        ksize = list(param.size())
        # to make [in_shape, out_shape, ksize, ksize]
        if len(ksize) > 1:
            ksize[0], ksize[1] = ksize[1], ksize[0]
        print(ksize)

    ##print(name, param.size())
'''
'''
# Keras style
print(3*"#-----------------------------#\n")
print("Summary from torchsummary")
summary(model, input_size=(3, 244, 244))
'''
