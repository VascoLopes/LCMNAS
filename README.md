# UMNAS

## Instalation:
sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
python3 -m venv venv
source venv/bin/activate
pip install wheel torchcontrib pytorchcv
pip install -r requirements.txt


## Get Data sets (informations are anonymized)
Download data.zip and extract it in the main directory of the code from https://drive.google.com/drive/folders/1bnXQfYfNyYKCSX05DSFG8gE---EkfXSP?usp=sharing

## Get Best Architectures Found (informations are anonymized)
Download models.zip (these do not have the autoaugment training) from https://drive.google.com/drive/folders/1bnXQfYfNyYKCSX05DSFG8gE---EkfXSP?usp=sharing


## Experiments (the results get automatically stored in the experiments folder)
### Mixed-performance estimation mechanism: (lambda=0.75)
python3 count.py --generations 50 --population 100 --dataset cifar10 --batch_size 96 —-mixed_training --mixed_fitness_lambda 0.75

### Using only zero-proxy estimation (lambda=1)
python3 count.py --generations 50 --population 100 --dataset cifar10 --batch_size 96 --without_training --mixed_fitness_lambda 1

### Using only regular training (lambda=0)
python3 count.py --generations 50 --population 100 --dataset cifar10 --batch_size 96 --mixed_fitness_lambda 0 

### Available datasets: cifar10 cifar100 ImageNet16-120
python3 count.py --generations 50 --population 100 --dataset cifar10 --batch_size 96 —-mixed_training --mixed_fitness_lambda 0.75

python3 count.py --generations 50 --population 100 --dataset cifar100 --batch_size 96 —-mixed_training --mixed_fitness_lambda 0.75

python3 count.py --generations 50 --population 100 --dataset ImageNet16 --batch_size 96 —-mixed_training --mixed_fitness_lambda 0.75 


### Train with auto-augment
python aux.py --searched_dataset cifar10 --dataset cifar10 --auto_augment --epochs 1500 --reset_weight --model_path "path_to_model"


### Transfer to other data set
python aux.py --searched_dataset cifar10 --dataset ImageNet16 --reset_weight --model_path "path_to_model"



Thank you for the time spent reviewing our paper.