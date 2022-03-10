import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

import data_loader_cifar10 as cifar10_ds
import data_loader_cifar100 as cifar100_ds
import data_loader_imagenet as imagenet_ds

import numpy as np
import copy
import time
import os
import sys, gc
from collections import OrderedDict 


# Name of all architectures implemented in pytorch
'''
model_names = sorted(name for name in models.__dict__
if name.islower() and not name.startswith("__")
and callable(models.__dict__[name]))
'''

class Utils():

    #--------------------------------------------------------------------------------#
    def __init__(self, batch_size, device='cuda'):
    # Parameters:
        self.batch_size   = batch_size
        self.device       = device
        
    #--------------------------------------------------------------------------------#
    def getCrossEntropyLoss(self):
        #print("[Using CrossEntropyLoss...]")
        criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        #criterion = nn.CrossEntropyLoss()

        return (criterion)

    #--------------------------------------------------------------------------------#
    def getSGDOptimizer(self, model, learningRate = 0.001, momentum=0.9, weight_decay=3e-4): #lr = 0.001
        #print("[Using small learning rate with momentum...]")
        optimizer_conv = optim.SGD(list(filter(
            lambda p: p.requires_grad, model.parameters())), lr=learningRate, momentum=momentum, weight_decay=weight_decay)

        return (optimizer_conv)

    #--------------------------------------------------------------------------------#
    def getLrScheduler(self, model, step_size=7, gamma=0.1):
        print("[Creating Learning rate scheduler...]")
        exp_lr_scheduler = lr_scheduler.StepLR(model, step_size=step_size, gamma=gamma)

        return (exp_lr_scheduler)
    
    #--------------------------------------------------------------------------------#
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    #--------------------------------------------------------------------------------#
    def get_train_dataset(self, args, data_dir:str="./data/", dataset:str="cifar10", \
                        datasetType:str="Partial", shuffle=False, \
                        auto_augment=False, resize=False, resizelength=300):
        # data_dir -> folder containing all datasets
        # dataset -> name of the specified dataset (and folder inside data_dir)
        data_dir = os.path.join(data_dir, dataset+'/')
        dataset = dataset.lower()
        if 'cifar100' in dataset:
            return cifar100_ds.get_train_valid_loader(data_dir, args.batch_size, datasetType=datasetType, \
                                            cutout = args.cutout, cutout_length=args.cutout_length, \
                                            shuffle=shuffle, auto_augment=auto_augment, resize=resize, resizelength=resizelength)
        elif 'cifar10' in dataset:
            return cifar10_ds.get_train_valid_loader(data_dir, args.batch_size, datasetType=datasetType, \
                                            cutout = args.cutout, cutout_length=args.cutout_length, \
                                            shuffle=shuffle, auto_augment=auto_augment, resize=resize, resizelength=resizelength)
        elif 'imagenet16' in dataset:
            return imagenet_ds.get_train_valid_loader('./data/ImageNet16/', args.batch_size, datasetType=datasetType, \
                                            cutout = args.cutout, cutout_length=args.cutout_length, \
                                            shuffle=shuffle, auto_augment=auto_augment, resize=resize, resizelength=resizelength)
        elif 'svhn' in dataset:
            pass

    def get_test_dataset(self, args, data_dir:str="./data/", dataset:str="cifar10", datasetType:str="Partial", resize = False, resizelength=300):
        data_dir = os.path.join(data_dir, dataset+'/')
        dataset = dataset.lower()
        if 'cifar100' in dataset:
            return cifar100_ds.get_test_loader(data_dir, args.batch_size, resize=resize, resizelength=resizelength)
        elif 'cifar10' in dataset:
            return cifar10_ds.get_test_loader(data_dir, args.batch_size, resize=resize, resizelength=resizelength)
        elif 'imagenet16' in dataset:
            return imagenet_ds.get_test_loader('./data/ImageNet16/', args.batch_size, resize=resize, resizelength=resizelength)
        elif 'svhn' in dataset:
            pass

    #--------------------------------------------------------------------------------#
    ''' Train Normal Models function '''
    def trainNormalModels(self, model, dataloader, criterion, optimizer, epoch=None):
        model.train()

        correct = 0
        total = 0
        correct_batch = 0
        total_batch = 0
        lossTotal = 0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = inputs

            # forward + backward + optimize
            outputs = model(outputs)

            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item() * inputs.size(0)
            lossTotal += running_loss # epoch loss
            correct_batch = (predicted == labels).sum().item()
            total_batch = labels.size(0)  

            if i % 500 == 0:    # print every 200 mini-batches
                print('[%5d] loss: %.5f ; Accuracy: %.2f'%
                    (i + 1, running_loss/total_batch, 100 * correct_batch / total_batch))

            running_loss = 0.0
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        lossTotal = lossTotal/total
        if epoch != None:
            print(f'Epoch {epoch} - Train Accuracy: {accuracy},    Loss: {lossTotal}')

        return model, accuracy, lossTotal
    #--------------------------------------------------------------------------------#
    ''' Validation For Normal Models function '''
    def evaluateNormalModels(self, model, valloader, criterion, epoch=None):
        model.eval()
        correct = 0
        total = 0
        running_loss = 0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc  = 100 * correct / total
        lossTotal = running_loss / total
        if epoch != None:
            print(f'Epoch {epoch} - Val Accuracy: {acc},    Loss: {loss}')
        return acc, lossTotal

    #--------------------------------------------------------------------------------#
    ''' Train function '''
    def train(self, model, dataloader, criterion, optimizer, epoch=None, verbose = False):
        losses = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        # train scheme
        model.train()

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = inputs
            # forward + backward + optimize
            for layer in model:
                #print(f'Input shape: {outputs.shape}')
                #print(f'Layer: {layer}')
                outputs = layer(outputs)
            

            loss = criterion(outputs, labels)
            loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            # Stats
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            n = inputs.size(0)

            losses.update(loss.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            # print stats
            if i % 200 == 0 and verbose:
                # print epoch batch/totalbatches loss lossavg top1 top1avg
                print('Epoch: [{0}][{1}/{2}]\t' 
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f}))'.format(
                        epoch, i, len(dataloader), loss=losses, top1=top1, top5=top5))

        # print epoch stats
        if epoch != None:
            print('Epoch {0} : ' 
            'Loss: {loss.val:.4f}\t\t' 
            'Acc@1: {top1.avg:.3f}\t\t'
            'Acc@5: {top5.avg:.3f}'.format(
                epoch, loss=losses, top1=top1, top5=top5))
            #print(f'Epoch {epoch} - Train Accuracy: {accuracy},    Loss: {lossTotal}')

        return model, top1, top5, losses

    #--------------------------------------------------------------------------------#
    ''' Validation function '''
    def evaluate(self, model, valloader, criterion, epoch=None):
        losses = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        
        # model scheme
        model.eval()

        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #outputs = model(inputs)

                outputs = inputs
                for layer in model:
                    outputs = layer(outputs)

                loss = criterion(outputs, labels)
                
                # Stats
                prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
                n = inputs.size(0)

                losses.update(loss.item(), n)
                top1.update(prec1.data.item(), n)
                top5.update(prec5.data.item(), n)
        
        if epoch != None:
            print('Epoch {0} : ' 
            'Loss: {loss.val:.4f}\t\t' 
            'Acc@1: {top1.avg:.3f}\t\t'
            'Acc@5: {top5.avg:.3f}'.format(
                epoch, loss=losses, top1=top1, top5=top5))
            #print(f'Epoch {epoch} - Val Accuracy: {acc},    Loss: {loss}')
        return top1, top5, losses

    #--------------------------------------------------------------------------------#
    ''' Test function '''
    def test(self, model, testloader):
        model.eval()
        correct = 0
        total = 0
        inference = []
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = inputs
                
                torch.cuda.synchronize()
                start_time = int(round(time.time()*1000))

                for layer in model:
                    outputs = layer(outputs)

                torch.cuda.synchronize()
                end_time = int(round(time.time()*1000))
                
                _, predicted = torch.max(outputs.data, 1)
                n = labels.size(0)
                total += n
                correct += (predicted == labels).sum().item()
                
                inference.append((end_time - start_time)/n)

        acc = 100 * correct / total
        return acc, inference
    
    # ----- Change classification output of pytorch models ------ #
    def change_CNN_classifier(self, model, model_name, n_classes):
        network = model_name.casefold()
        if 'vgg' in network or 'alexnet' in network:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

        elif 'resnet' in network or 'resnext' in network or 'googlenet' in network or 'shufflenet' in network:
            model.fc = nn.Linear(model.fc.in_features, n_classes)

        elif 'densenet' in network:
            model.classifier = nn.Linear(model.classifier.in_features, n_classes)

        elif 'squeezenet' in network:
            model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, n_classes, model.classifier[1].kernel_size)
            
        elif 'mnasnet' in network or 'mobilenet' in network:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

        elif 'bit' in network:
            model.head[-1] = nn.Identity()


        return model
    
    #--------------------------------------------------------------------------------#
    def add_bn(self, model, after_activation=True):
        new_model = []
        #[('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 2}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (3, 3), 'stride': 1}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 1}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 1}), ('Conv2d', {'out_channels': 512, 'kernel_size': (1, 1)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 256, 'kernel_size': (3, 3), 'stride': 1}), ('Conv2d', {'out_channels': 1024, 'kernel_size': (1, 1)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Conv2d', {'out_channels': 2048, 'kernel_size': (3, 3)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 1024, 'kernel_size': (1, 1), 'stride': 1}), ('Conv2d', {'out_channels': 48, 'kernel_size': (3, 3)}), ('BatchNorm2d', {}), ('ReLU', {}), ('MaxPool2d', {'kernel_size': 2}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 24, 'kernel_size': (1, 1), 'stride': 1}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 2}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (3, 3), 'stride': 1}), ('Conv2d', {'out_channels': 256, 'kernel_size': (1, 1)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 512, 'kernel_size': (1, 1), 'stride': 2}), ('Dropout', {'p': 0.5}), ('Linear', {'out_features': 10})]
        for idx,layer in enumerate(model):
            new_model.append(layer)
            try:
                if 'batchnorm' in model[idx+1][0].lower():
                    continue
                if 'elu' in model[idx+1][0].lower() or 'soft' in model[idx+1][0].lower(): #activation functions
                    if after_activation == False:
                        continue
                new_model.append(('BatchNorm2d', {}))
            except:
                pass
        return (new_model)

    #--------------------------------------------------------------------------------#
    def get_size(self, obj):
        marked = {id(obj)}
        obj_q = [obj]
        sz = 0

        while obj_q:
            sz += sum(map(sys.getsizeof, obj_q))

            # Lookup all the object referred to by the object in obj_q.
            # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
            all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

            # Filter object that are already marked.
            # Using dict notation will prevent repeated objects.
            new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

            # The new obj_q will be the ones that were not marked,
            # and we will update marked with their ids so we will
            # not traverse them again.
            obj_q = new_refr.values()
            marked.update(new_refr.keys())

        return sz

    #--------------------------------------------------------------------------------#
    def clean_memory(self):
        # Free Torch memory & Call Garbage collector
        try:
            gc.collect()
        except:
            pass
        '''
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            pass
        '''

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# To store information about predictions
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    #print(correct.view(-1))
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res



# Tests
if __name__ == "__main__":
    a = [('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 2}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (3, 3), 'stride': 1}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 1}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 1}), ('Conv2d', {'out_channels': 512, 'kernel_size': (1, 1)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 256, 'kernel_size': (3, 3), 'stride': 1}), ('Conv2d', {'out_channels': 1024, 'kernel_size': (1, 1)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Conv2d', {'out_channels': 2048, 'kernel_size': (3, 3)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 1024, 'kernel_size': (1, 1), 'stride': 1}), ('Conv2d', {'out_channels': 48, 'kernel_size': (3, 3)}), ('BatchNorm2d', {}), ('ReLU', {}), ('MaxPool2d', {'kernel_size': 2}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 24, 'kernel_size': (1, 1), 'stride': 1}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (1, 1), 'stride': 2}), ('Bottleneck', {'out_channels': 244, 'kernel_size': (3, 3), 'stride': 1}), ('Conv2d', {'out_channels': 256, 'kernel_size': (1, 1)}), ('BatchNorm2d', {}), ('ReLU', {}), ('Bottleneck', {'out_channels': 512, 'kernel_size': (1, 1), 'stride': 2}), ('Dropout', {'p': 0.5}), ('Linear', {'out_features': 10})]
    utils = Utils(10)
    
    newmodel = utils.add_bn(a)
    
    print(a)
    print("----")
    print(newmodel)