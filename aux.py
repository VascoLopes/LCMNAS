from operations import LinearLayer
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
import utils, count
import argparse
import numpy as np
import time, copy




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

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


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
    parser.add_argument('--mixed_fitness_lambda', type=float, default=0.5, help='percentage of importance given to without train (1-lambda for val)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--population', type=int, default=25, help='population size')
    parser.add_argument('--generations', type=int, default=50, help='number of generations')
    parser.add_argument('--max_neurons', type=int, default=6000, help="number of max input neurons in final MLP layers")
    parser.add_argument('--prob_residual', type=int, default=50, help='probability of conv becomming residual block')
    parser.add_argument('--prob_residual_downsample', type=int, default=30, help='probability of residual block includes downsampling')
    parser.add_argument('--init_pop', action='store_false', default=True, help='init population defined') 
    parser.add_argument('--cutout', action='store_true', default=True, help='use cutout') 
    parser.add_argument('--auto_augment', action='store_true', default=False, help='use autoaugment')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--random_search_layer', action='store_true', default=False, help='apply random search on layer selection (baseline)')
    parser.add_argument('--random_search_model', action='store_true', default=False, help='apply random search on model selection (baseline)')
    parser.add_argument('--prunning', action='store_false', default=True, help='use prune in final training') #TODO true?
    parser.add_argument('--prunning_threshold', type=float, default=1e-2, help='threshold value for prunning (abs(x)>threshold)') #TODO true?
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--searched_dataset', type=str, default='cifar10', help='use model search on dataset X')
    parser.add_argument('--reset_weight', action='store_true', default=False, help='reset weights of the model') 
    parser.add_argument('--model_path', type=str, default='', help='directory of the model')
    parser.add_argument('--resize_length', type=int, default=32)

    parser.add_argument('--elitism', type=int, default=int((0.15*parser.parse_args().population) if int(0.15*parser.parse_args().population > 0) else 1), help="number of elements to pass through elitism")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils = utils.Utils(96,device)

    if args.searched_dataset=="cifar10":
        if args.model_path != "":
            model = torch.load(args.model_path)
        else:
            model = torch.load('./best_models/model_cifar10_lambda075.pth')
    elif args.searched_dataset=="cifar100":
        if args.model_path != "":
            model = torch.load(args.model_path)
        else:
            model = torch.load('./best_models/model_cifar100_lambda05.pth')
    else:
        if args.model_path != "":
            model = torch.load(args.model_path)
        else:
            model = torch.load('./best_models/model_imagenet16_lamda075.pth')

    print (f'\n### Model searched on {args.searched_dataset} now training on {args.dataset} [...] ###')
    # Train model


    if args.searched_dataset=="cifar10" or args.searched_dataset=="cifar100":
        trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=args.data_dir, \
                                                                    dataset=args.dataset, datasetType='TrainEntire', shuffle=True, \
                                                                    auto_augment=args.auto_augment, resize=True, resizelength=args.resize_length)
    else:
        trainloader, valloader, n_classes = utils.get_train_dataset(args, data_dir=args.data_dir, \
                                                                    dataset=args.dataset, datasetType='TrainEntire', shuffle=True, \
                                                                    resize=True, auto_augment=args.auto_augment, resizelength=16)


    model[-1] = LinearLayer(model[-1].op.in_features, n_classes)
    model.to(device)
    #print(model[-1].op.weight)
    if args.reset_weight:
        model.apply(weight_reset)
    #print(model[-1].op.weight)
    model, best_model_wts, best_model_lowest_loss, all_train_acc, all_train_loss, all_val_acc, \
                all_val_loss, best_acc_val, best_loss_val, endTimeFinalTrain \
                = train_network(model, utils, args, trainloader, valloader)

    model.load_state_dict(best_model_wts)

    if args.searched_dataset=="cifar10" or args.searched_dataset=="cifar100":
        testloader = utils.get_test_dataset(args, data_dir=args.data_dir, dataset=args.dataset)
    else: #IMGNET -> CIFAR
        testloader = utils.get_test_dataset(args, data_dir=args.data_dir, dataset=args.dataset, resize=True, resizelength=16)

    accuracy_test, inference_time = utils.test(model, testloader)

    print(f'trainAcc={all_train_acc}')
    print(f'trainLoss={all_train_loss}')
    print(f'valAcc={all_val_acc}')
    print(f'valLoss={all_val_loss}')
    print(f'Final Test Accuracy: {accuracy_test} : val accuracy {best_acc_val[0]} - train accuracy {all_train_acc[best_acc_val[1]]}')
    print(f'Time taken to train the final model from scratch: {endTimeFinalTrain}')

    try:
        model.eval()
        print(f'Model parameters:{utils.count_parameters(model)}')
        print(f'Inference Time Mean: {np.mean(inference_time)}, STD:{np.std(inference_time)} ms')
    except:
        pass
    torch.save(model, "./best_models/model_searched_"+str(args.searched_dataset)+"trainedon_"+str(args.dataset)+".pth")
