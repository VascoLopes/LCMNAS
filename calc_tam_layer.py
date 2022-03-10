import numpy as np
import math
from functools import reduce 

def Conv_Padding(kernel_size=[3,3], img_sizes=[32,32], stride=[1], padding=[0], dilation=[1]):
    img_paddings = []
    for idx, size in enumerate(img_sizes): # [0] -> height; [1] -> width; [2] maybe depth (3d conv)
        try:
            if len(kernel_size) > 1: #kernel_size is a list with same len of img_size
                img_paddings += [math.ceil(((size - 1) * (stride[0] - 1) + dilation[0] * (kernel_size[idx] - 1)) / 2)]
            else:
                img_paddings += [math.ceil(((size - 1) * (stride[0] - 1) + dilation[0] * (kernel_size - 1)) / 2)]
        except:
            img_paddings += [math.ceil(((size - 1) * (stride[0] - 1) + dilation[0] * (kernel_size - 1)) / 2)]
        #width_padding  = math.ceil(((width - 1) * (stride[0] - 1) + dilation[0] * (kernel_size[0] - 1)) / 2)
    #height_padding = np.floor((height+2*padding[1]-dilation[0]*(kernel_size[1]-1)-1)/stride[0] + 1)
    #width_padding  = np.floor((width+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0] + 1)
    return tuple(img_paddings)


def Conv_Output(kernel_size=3, size=32, stride=1, padding=0):
    output = (size - kernel_size + 2*padding)/stride + 1
    return output

def Pool_Stride(kernel_size=3, img_size=[32,32]):
    return

def MaxPool2d(kernel_size, stride=None, padding=[0], dilation=[1], return_indices=False, ceil_mode=False):
    if stride == None:
        stride = kernel_size
    Hout = np.floor( (Hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1 ) / stride[0]+1)
    return int(Hout)


def ConvTranspose2d(in_channels, out_channels, kernel_size, stride=[1], padding=[0], dilation=[1], groups=1, bias=True):
    Hout=np.floor((Hin-1)*stride[0]-2*padding[0]+kernel_size[0])
    return int(Hout)


def Flatten(img_size, channels = 3):
    flattenSum = reduce((lambda x, y: x * y), img_size)
    flattenSum *= channels
    return (flattenSum)


def NNtest():
    global Hin
    '''
    Hin = 32

    Hin = Conv2d(4, 32, [5],[1],[2])
    print(Hin)

    Hin = MaxPool2d([2],[2])
    print(Hin)

    Hin = Conv2d(32, 64, [5],[1],[2])
    print(Hin)

    Hin = Flatten(Hin, Hin, 64)
    print(Hin)
    '''
    Hin = 224

    Hin = MaxPool2d([2])
    print(Hin)
    #Hin = ConvTranspose2d(64,64,[2],[2],[0])
    #print(Hin)

    Hin = Conv2d(64,64,[7],[1],[3])
    print(Hin)

    Hin = Conv2d(32,4,[5],[1],[2])
    print(Hin)

# ----------------------------------------------------------------------
# 				MAIN
# ----------------------------------------------------------------------
#NNtest()
