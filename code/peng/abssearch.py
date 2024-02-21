import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import copy
import numpy as np
import time
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--max_iters', type=int, default=20)
parser.add_argument('--model_cache', type=str, default='./checkpoint/ckpt.pth', help='model to be loaded')
parser.add_argument(
    '--max_cycle',
    type=int,
    default=5,
    help='Search for best pruning plan times. default:10'
)

parser.add_argument(
    '--max_preserve',
    type=int,
    default=9,
    help='Minimum percent of prune per layer'
)

parser.add_argument(
    '--preserve_type',
    type = str,
    default = 'layerwise',
    help = 'The preserve ratio of each layer or the preserve ratio of the entire network'

)

parser.add_argument(
    '--food_number',
    type=int,
    default=10,
    help='Food number'
)

parser.add_argument(
    '--food_dimension',
    type=int,
    default=13,
    help='Food dimension: num of conv layers. default: vgg16->13 conv layer to be pruned'
)    

parser.add_argument(
    '--food_limit',
    type=int,
    default=5,
    help='Beyond this limit, the bee has not been renewed to become a scout bee'
)

parser.add_argument(
    '--honeychange_num',
    type=int,
    default=2,
    help='Number of codes that the nectar source changes each time'
)

parser.add_argument(
    '--best_honey',
    type=int,
    nargs='+',
    default=None,
    help='If this hyper-parameter exists, skip bee-pruning and fine-tune from this prune method'
)

parser.add_argument(
    '--best_honey_s',
    type=str,
    default=None,
    help='Path to the best_honey'
)

parser.add_argument(
    '--best_honey_past',
    type=int,
    nargs='+',
    default=None,
)

args = parser.parse_args()
food_dimension = 13
model = VGG('VGG11')

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Loading Model..')

width_list = [0.25, 0.5, 0.75, 1]
model = VGG('VGG11')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

if os.path.exists(args.model_cache):
    print('loading checkpoint {} ..........'.format(args.model_cache))
    checkpoint = torch.load(args.model_cache)
    model.load_state_dict(checkpoint['net'])
    print("loaded checkpoint {} epoch = {}" .format(args.model_cache, checkpoint['epoch']))

criterion = nn.CrossEntropyLoss()  # 分类问题使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)  # 优化器选择
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def find_nearest_index(values, targets):
    # 将目标值转换为索引（0到len(targets)-1）
    target_indices = {target: idx for idx, target in enumerate(targets)}
    
    def get_nearest_index(value):
        # 计算给定值与目标数组中每个值的差的绝对值
        differences = [abs(value - target) for target in targets]
        # 找到最小差的索引
        nearest_index = differences.index(min(differences))
        # 返回最接近值的索引
        return target_indices[targets[nearest_index]]
    
    # 映射每个输入值到其最接近的目标值的索引
    return [get_nearest_index(value/10) for value in values]

# Training
def train(epoch):
    global last_id
    print('\nEpoch: %d' % epoch)

    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

#Define BeeGroup 
class BeeGroup():
    """docstring for BeeGroup"""
    def __init__(self):
        super(BeeGroup, self).__init__() 
        self.code = [] #size : num of conv layers  value:{1,2,3,4,5,6,7,8,9,10}
        self.fitness = 0
        self.rfitness = 0 
        self.trail = 0

#Initilize global element
best_honey = BeeGroup()
NectraSource = []
EmployedBee = []
OnLooker = []
best_honey_state = {}


#Calculate fitness of a honey source
def calculationFitness(code, args):
    global best_honey
    global best_honey_state
    i = 0 
    width_can = find_nearest_index(code, width_list)

    for n, m in model.named_modules():
        width_id = width_can[i]
        if isinstance(m, nn.Conv2d):
            setattr(m, 'width_mult', width_list[width_id])
            last_id = width_id
            i += 1
        elif isinstance(m, nn.BatchNorm2d):
            setattr(m, 'width_id', last_id)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if correct > best_honey.fitness:
        best_honey_state = copy.deepcopy(model.module.state_dict())
        best_honey.code = copy.deepcopy(code)
        best_honey.fitness = correct

    return correct

#Initilize Bee-Pruning
def initilize():
    print('==> Initilizing Honey_model..')
    global best_honey, NectraSource, EmployedBee, OnLooker

    for i in range(args.food_number):
        NectraSource.append(copy.deepcopy(BeeGroup()))
        EmployedBee.append(copy.deepcopy(BeeGroup()))
        OnLooker.append(copy.deepcopy(BeeGroup()))
        for j in range(food_dimension):
            NectraSource[i].code.append(copy.deepcopy(random.randint(1,args.max_preserve)))

        #initilize honey souce
        NectraSource[i].fitness = calculationFitness(NectraSource[i].code, args)
        NectraSource[i].rfitness = 0
        NectraSource[i].trail = 0

        #initilize employed bee  
        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)
        EmployedBee[i].fitness=NectraSource[i].fitness 
        EmployedBee[i].rfitness=NectraSource[i].rfitness 
        EmployedBee[i].trail=NectraSource[i].trail

        #initilize onlooker 
        OnLooker[i].code = copy.deepcopy(NectraSource[i].code)
        OnLooker[i].fitness=NectraSource[i].fitness 
        OnLooker[i].rfitness=NectraSource[i].rfitness 
        OnLooker[i].trail=NectraSource[i].trail

    #initilize best honey
    best_honey.code = copy.deepcopy(NectraSource[0].code)
    best_honey.fitness = NectraSource[0].fitness
    best_honey.rfitness = NectraSource[0].rfitness
    best_honey.trail = NectraSource[0].trail

#Send employed bees to find better honey source
def sendEmployedBees():
    global NectraSource, EmployedBee
    for i in range(args.food_number):
        
        while 1:
            k = random.randint(0, args.food_number-1)
            if k != i:
                break

        EmployedBee[i].code = copy.deepcopy(NectraSource[i].code)

        param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
        R = np.random.uniform(-1, 1, args.honeychange_num)
        for j in range(args.honeychange_num):
            EmployedBee[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
            if EmployedBee[i].code[param2change[j]] < 1:
                EmployedBee[i].code[param2change[j]] = 1
            if EmployedBee[i].code[param2change[j]] > args.max_preserve:
                EmployedBee[i].code[param2change[j]] = args.max_preserve

        EmployedBee[i].fitness = calculationFitness(EmployedBee[i].code, args)

        if EmployedBee[i].fitness > NectraSource[i].fitness:                
            NectraSource[i].code = copy.deepcopy(EmployedBee[i].code)              
            NectraSource[i].trail = 0  
            NectraSource[i].fitness = EmployedBee[i].fitness 
            
        else:          
            NectraSource[i].trail = NectraSource[i].trail + 1

#Calculate whether a Onlooker to update a honey source
def calculateProbabilities():
    global NectraSource
    
    maxfit = NectraSource[0].fitness

    for i in range(1, args.food_number):
        if NectraSource[i].fitness > maxfit:
            maxfit = NectraSource[i].fitness

    for i in range(args.food_number):
        NectraSource[i].rfitness = (0.9 * (NectraSource[i].fitness / maxfit)) + 0.1

#Send Onlooker bees to find better honey source
def sendOnlookerBees():
    global NectraSource, EmployedBee, OnLooker
    i = 0
    t = 0
    while t < args.food_number:
        R_choosed = random.uniform(0,1)
        if(R_choosed < NectraSource[i].rfitness):
            t += 1

            while 1:
                k = random.randint(0, args.food_number-1)
                if k != i:
                    break
            OnLooker[i].code = copy.deepcopy(NectraSource[i].code)

            param2change = np.random.randint(0, food_dimension-1, args.honeychange_num)
            R = np.random.uniform(-1, 1, args.honeychange_num)
            for j in range(args.honeychange_num):
                OnLooker[i].code[param2change[j]] = int(NectraSource[i].code[param2change[j]]+ R[j]*(NectraSource[i].code[param2change[j]]-NectraSource[k].code[param2change[j]]))
                if OnLooker[i].code[param2change[j]] < 1:
                    OnLooker[i].code[param2change[j]] = 1
                if OnLooker[i].code[param2change[j]] > args.max_preserve:
                    OnLooker[i].code[param2change[j]] = args.max_preserve

            OnLooker[i].fitness = calculationFitness(OnLooker[i].code, args)

            if OnLooker[i].fitness > NectraSource[i].fitness:                
                NectraSource[i].code = copy.deepcopy(OnLooker[i].code)              
                NectraSource[i].trail = 0  
                NectraSource[i].fitness = OnLooker[i].fitness 
            else:          
                NectraSource[i].trail = NectraSource[i].trail + 1
        i += 1
        if i == args.food_number:
            i = 0

#If a honey source has not been update for args.food_limiet times, send a scout bee to regenerate it
def sendScoutBees():
    global  NectraSource, EmployedBee, OnLooker
    maxtrailindex = 0
    for i in range(args.food_number):
        if NectraSource[i].trail > NectraSource[maxtrailindex].trail:
            maxtrailindex = i
    if NectraSource[maxtrailindex].trail >= args.food_limit:
        for j in range(food_dimension):
            R = random.uniform(0,1)
            NectraSource[maxtrailindex].code[j] = int(R * args.max_preserve)
            if NectraSource[maxtrailindex].code[j] == 0:
                NectraSource[maxtrailindex].code[j] += 1
        NectraSource[maxtrailindex].trail = 0
        NectraSource[maxtrailindex].fitness = calculationFitness(NectraSource[maxtrailindex].code,  args )
 
 #Memorize best honey source
def memorizeBestSource():
    global best_honey, NectraSource
    for i in range(args.food_number):
        if NectraSource[i].fitness > best_honey.fitness:
            #print(NectraSource[i].fitness, NectraSource[i].code)
            #print(best_honey.fitness, best_honey.code)
            best_honey.code = copy.deepcopy(NectraSource[i].code)
            best_honey.fitness = NectraSource[i].fitness


def main():
    start_epoch = 0
    best_acc = 0.0
    code = []

    start_time = time.time()
    
    bee_start_time = time.time()
    
    print('==> Start BeePruning..')

    initilize()

    #memorizeBestSource()

    for cycle in range(args.max_cycle):

        current_time = time.time()
        print(
            'Search Cycle [{}]\t Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime {:.2f}s\n'
            .format(cycle, best_honey.code, float(best_honey.fitness), (current_time - start_time))
        )
        start_time = time.time()

        sendEmployedBees() 
            
        calculateProbabilities() 
            
        sendOnlookerBees()  

        sendScoutBees() 
                

    print('==> BeePruning Complete!')
    bee_end_time = time.time()
    print(
        'Best Honey Source {}\tBest Honey Source fitness {:.2f}%\tTime Used{:.2f}s\n'
        .format(best_honey.code, float(best_honey.fitness), (bee_end_time - bee_start_time))
    )
        #checkpoint.save_honey_model(state)

    code = best_honey.code
    width_can = find_nearest_index(code, width_list)
    #transfer code to int
    i = 0 
    for n, m in model.named_modules():
        width_id = width_can[i]
        if isinstance(m, nn.Conv2d):
            setattr(m, 'width_mult', width_list[width_id])
            last_id = width_id
            i += 1
        elif isinstance(m, nn.BatchNorm2d):
            setattr(m, 'width_id', last_id)
   
    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)  # 训练一次更新完一次参数
        test(epoch)  # 就用更新完的参数测试一次
        scheduler.step()
       

if __name__ == '__main__':
    main()