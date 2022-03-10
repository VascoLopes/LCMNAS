import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#OPS['Conv2d'](Cin, Cout, kernel_size, padding)
OPS = {
  'Conv2d'      : lambda Cin, Cout, kernel_size, padding: nn.Conv2d(Cin, Cout, kernel_size, padding=padding),#Conv2D(Cin, Cout, kernel_size, padding),
  'Linear'      : lambda in_features, out_features, unflatten: LinearLayer(in_features, out_features, unflatten),
  'BatchNorm2d' : lambda features : nn.BatchNorm2d(features), #nn.Sequential(nn.BatchNorm2d(features))
  'Dropout'     : lambda p : nn.Dropout(p),
  'Dropout2d'   : lambda p : nn.Dropout2d(p),
  'AvgPool2d'   : lambda kernel_size : nn.AvgPool2d(kernel_size),
  'MaxPool2d'   : lambda kernel_size : nn.AvgPool2d(kernel_size),
  'AdaptiveAvgPool2d' : lambda out_features : nn.AdaptiveAvgPool2d(out_features),
  'AdaptiveMaxPool2d' : lambda out_features : nn.AdaptiveMaxPool2d(out_features),
  'ReLU'        : lambda inplace : nn.ReLU(inplace),
  'ReLU6'       : lambda inplace : nn.ReLU(inplace),
  'SELU'        : lambda inplace : nn.SELU(inplace),
  'ELU'         : lambda inplace : nn.ELU(inplace),
  'Bottleneck'  : lambda in_channels, planes, stride, downsample, groups, base_width, dilation, norm_layer: Bottleneck(in_channels, planes, stride, downsample, groups, base_width, dilation, norm_layer)
}

class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, unflatten=False):
        super(LinearLayer, self).__init__()
        self.unflatten = unflatten
        # The output shape can be calculated here and put into an inner variable
        self.op = nn.Linear(in_features=in_features, out_features=out_features)#nn.Sequential(
            #nn.Linear(in_features=in_features, out_features=out_features)
        #)

    def forward(self, x, unflatten=False):
        #print (f'Shape:{x.shape}')
        #x = torch.flatten(x)
        shape = x.shape
        if (len(shape) > 2): #has more than 2 dimensions (channels, dim, _)
            #print("entrou")
            #x = torch.flatten(x, 1)
            x = x.view(shape[0], -1) # This could also be a "class Flatten()..."
            #print (f'Shape:{x.shape}')        
        if self.unflatten:
            return (self.op(x).view(shape)) # Back to the same shape.
        return self.op(x) #or return the out_features


class Conv2D(nn.Module):

    def __init__(self, Cin, Cout, kernel_size=(3,3), padding=(1,1)):
        super(Conv2D, self).__init__()
        self.op = nn.Conv2d(Cin, Cout, kernel_size, padding=padding)#nn.Sequential(
            #nn.Conv2d(Cin, Cout, kernel_size, padding=padding),
        #)

    def forward(self, x):
        return self.op(x)

# ------ Residual blocks ------
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        if stride != 1 or inplanes != planes * self.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )
        self.output = planes * self.expansion
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #print(x.shape)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        #print(out.shape)
        #print(identity.shape)
        out += identity
        out = self.relu(out)

        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



# generate input sample and forward to get shape
# n_size = self._get_conv_output(input_shape) # (can be used inside init of a model)
def _get_conv_output(self, shape):
    bs = 1
    input = Variable(torch.rand(bs, *shape))
    output_feat = self._forward_features(input)
    n_size = output_feat.data.view(bs, -1).size(1)
    return n_size
# forward the conv to get output size
def _forward_features(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    return



def transformLayerNameIntoPytorch(layer, in_channels=None, out_channels=None, kernel_size=None, in_features=None, out_features=None, p=None, num_features=None):
    if layer == "Linear":
        return (nn.Linear(in_features=in_features, out_features=out_features))
    if layer == "Conv1d":
        return (nn.Conv1d(in_channels,out_channels,kernel_size))
    if layer == "Conv2d":
        return (nn.Conv2d(in_channels,out_channels,kernel_size))
    if layer == "Conv3d":
        return (nn.Conv3d(in_channels,out_channels,kernel_size))
    if layer == "BatchNorm1d":
        return (nn.BatchNorm1d(num_features))
    if layer == "BatchNorm2d":
        return (nn.BatchNorm2d(num_features))
    if layer == "BatchNorm3d":
        return (nn.BatchNorm3d(num_features))
    if layer == "MaxPool1d":
        return (nn.MaxPool1d(kernel_size))
    if layer == "MaxPool2d":
        return (nn.MaxPool2d(kernel_size))
    if layer == "MaxPool3d":
        return (nn.MaxPool3d(kernel_size))
    if layer == "AvgPool1d":
        return (nn.AvgPool1d(kernel_size))
    if layer == "AvgPool2d":
        return (nn.AvgPool2d(kernel_size))
    if layer == "AvgPool3d":
        return (nn.AvgPool3d(kernel_size))
    if layer == "AdaptiveAvgPool1d":
        return (nn.AdaptiveAvgPool1d(out_features))
    if layer == "AdaptiveAvgPool2d":
        return (nn.AdaptiveAvgPool2d(out_features))
    if layer == "AdaptiveAvgPool3d":
        return (nn.AdaptiveAvgPool3d(out_features))
    if layer == "AdaptiveMaxPool1d":
        return (nn.AdaptiveMaxPool1d(out_features))
    if layer == "AdaptiveMaxPool2d":
        return (nn.AdaptiveMaxPool2d(out_features))
    if layer == "AdaptiveMaxPool3d":
        return (nn.AdaptiveMaxPool3d(out_features))
    if layer == "ReLU":
        return (nn.ReLU())
    if layer == "ReLU6":
        return (nn.ReLU6())
    if layer == "SELU":
        return (nn.SELU())
    if layer == "ELU":
        return (nn.ELU())
    if layer == "Dropout":
        return (nn.Dropout(p))
    if layer == "Dropout2d":
        return (nn.Dropout2d(p))
    if layer == "Dropout3d":
        return (nn.Dropout3d(p))