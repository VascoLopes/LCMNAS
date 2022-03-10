import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
import gc

from operations import *

# Evaluation Without Training
def get_batch_jacobian(net, x, target, generated=False):
	net.zero_grad()
	x.requires_grad_(True)

	if generated:
		y=x
		for layer in net:
			y = layer(y)
	else:
		y = net(x)

	y.backward(torch.ones_like(y))
	jacob = x.grad.detach()
	#print(model)
	#print(np.sum(model.output.weight.grad.detach().cpu().numpy()))

	del x, y
	return jacob, target.detach()

def maximal_entropy_random_walks(corrs, v):
    return np.sum(corrs*v/np.sum(corrs*v))

def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    #print(corrs.shape)
    v, _  = np.linalg.eig(corrs)
    #print (v.shape)
    k = 1e-5
    #print((np.sum(np.log(corrs+1))))
    return 1e4/np.sum(np.log(v + k) + 1./(v + k))


def zero_cost_proxy(train_loader, model, device, batch_size, desired_size=256, generated=False):
	data_iterator = iter(train_loader)
	iterations = np.int(np.ceil(desired_size/batch_size))
	jacobs = []

	for i in range(iterations):
		x, target = next(data_iterator)
		x, target = x.to(device), target.to(device)
		#print(x.shape)

		jacobs_batch, _= get_batch_jacobian(model, x, target, generated)
		#print(i)
		jacobs.append(jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy())
	jacobs = np.concatenate(jacobs, axis=0)

	# if the np.ceil has not perfectly rounded and increased the iterations by 1
	if(jacobs.shape[0]>desired_size):
		jacobs = jacobs[0:desired_size, :]

	s = eval_score(jacobs)

	del data_iterator, iterations, jacobs

	return (s)


# Tests
if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	mean = [x / 255 for x in [125.3, 123.0, 113.9]]
	std  = [x / 255 for x in [63.0, 62.1, 66.7]]
		
	lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
	train_transform = transforms.Compose(lists)
	test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
	train_data = dset.CIFAR10 (".", train=True , transform=train_transform, download=True)
	batch_size = 256
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


	arch = ['densenet40_k12_cifar10', 'seresnet20_cifar10', 'resnet56_cifar10', 'pyramidnet110_a84_cifar10', 'wrn16_10_cifar10']
	for network in arch:

		data_iterator = iter(train_loader)
		x, target = next(data_iterator)
		x, target = x.to(device), target.to(device)

		model = ptcv_get_model(network)
		model.output = nn.Linear(model.output.in_features, 1)
		model = model.to(device)
		jacobs, labels= get_batch_jacobian(model, x, target)
		#print(jacobs.shape)
		jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
		#print (jacobs.shape)
		s = eval_score(jacobs)
		print(f'{network:30} score:{s}')
		gc.collect()
		
	model = ptcv_get_model('resnet56_cifar10')
	model.output = nn.Linear(4096, 1)
	model = model.to(device)
		
	data_iterator = iter(train_loader)
	x, target = next(data_iterator)
	x, target = x.to(device), target.to(device)
	s = zero_cost_proxy(train_loader, model, 'cuda', batch_size)
	#jacobs, labels= get_batch_jacobian(model, x, target)
	#jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
	#s = eval_score(jacobs)
	print(f'SimpleCNN                      score:{s}')
	'''
	resnet18 = torchvision.models.resnet18()
	resnet18.fc = nn.Linear(512,1)
	resnet18 = resnet18.to(device)

	jacobs, labels= get_batch_jacobian(resnet18, x, target, 1, device, "")
	jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
	s = eval_score(jacobs, labels)
	print(f'Resnet18 score:{s}')

	#-----#
	data_iterator = iter(train_loader)
	x, target = next(data_iterator)
	x, target = x.to(device), target.to(device)

	net = ptcv_get_model("seresnet20_cifar10")
	net.output = nn.Linear(64,1)
	net = net.to(device)

	jacobs, labels= get_batch_jacobian(net, x, target, 1, device, "")
	jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
	s = eval_score(jacobs, labels)
	print(f'SeResNet20 score:{s}')
	'''





