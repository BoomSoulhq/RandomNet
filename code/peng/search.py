import pickle
import torch.utils
import torch.utils.data.distributed
import torchvision
from thop import profile
import sys

sys.path.append("../../")
from utils import *
import argparse
from torchvision import datasets, transforms
import models


channel_scale = []
for i in range(31):
    channel_scale += [(10 + i * 3) / 100]

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--max_iters', type=int, default=20)
parser.add_argument('--net_cache', type=str, default='./checkpoint/ckpt.pth', help='model to be loaded')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--save_dict_name', type=str, default='save_dict.txt')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                     help='number of data loading workers (default: 4)')
args = parser.parse_args()

layer_num = 8

max_FLOPs = 2050
min_FLOPs = 1950

# file for save the intermediate searched results
save_dict = {}
if os.path.exists(args.save_dict_name):
    f = open(args.save_dict_name, 'rb')
    save_dict = pickle.load(f)
    f.close()
    print(save_dict, flush=True)

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



def num_change(x: float):
    if x < 0 or x > 1:
        return "Input Error!"
    standard = [0.25, 0.5, 0.75, 1]
    for i in range(len(standard) - 1):
        if x == standard[i] or x == standard[-1]:
            return x
        elif x < standard[0]:
            return standard[0]
        else:
            left = x - standard[i]
            right = standard[i + 1] - x
            if left > 0 and right > 0:
                if left <= right:
                    return standard[i]
                else:
                    return standard[i + 1]
            else:
                continue


# infer the accuracy of a selected pruned net (identidyed with ids)
def infer(model, criterion, ids, val_loader):  # ids==can

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # evaluate the corresponding pruned network
    i = 0
    width_list = [0.25, 0.5, 0.75, 1]
    for n, m in model.named_modules():
        width_id = ids[i]
        # 稀疏率近似
        if isinstance(m, nn.Conv2d):
            setattr(m, 'width_mult', num_change(width_list[width_id]))
            last_id = width_id  # 改成近似后的
            i += 1
        elif isinstance(m, nn.BatchNorm2d):
            setattr(m, 'width_id', last_id)

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images, ids.astype(np.int32))
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'

              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


# prepare ids for testing
def test_candidates_model(model, criterion, candidates, cnt, test_dict,save_dict):
    for can in candidates:
        print('test {}th model'.format(cnt), flush=True)
        print(list(can[:-1].astype(np.int32)))
        print('FLOPs = {:.2f}M'.format(can[-1]), flush=True)

        t_can = tuple(can[:-1])
        assert t_can not in test_dict.keys()
        print(t_can, flush=True)

        if t_can in save_dict.keys():
            Top1_err = save_dict[t_can]
            print('Already tested Top1_err = {:.2f}'.format(Top1_err))

        else:
            Top1_acc, Top5_acc, loss = infer(model, criterion, can[:-1])
            Top1_err = 100.0 - Top1_acc
            Top5_err = 100.0 - Top5_acc
            print('Top1_err = {:.2f} Top5_err = {:.2f} loss = {:.4f}'.format(Top1_err, Top5_err, loss), flush=True)
            save_dict[t_can] = Top1_err
        cnt += 1
        assert Top1_err >= 0
        can[-1] = Top1_err
        test_dict[t_can] = can[-1]

    return candidates, cnt


# mutation operation in evolution algorithm
def get_mutation(keep_top_k, layer_num, mutation_num, m_prob, test_dict, untest_dict, model,channel_scale,max_FLOPs,min_FLOPs):
    print('mutation ......', flush=True)
    res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10
    while len(res) < mutation_num and iter < max_iters:
        ids = np.random.choice(k, mutation_num)
        select_seed = np.array([keep_top_k[id] for id in ids])
        is_m = np.random.choice(np.arange(0, 2), (mutation_num, layer_num + 1), p=[1 - m_prob, m_prob])
        mu_val = np.random.choice(np.arange(1, len(channel_scale)), (mutation_num, layer_num + 1)) * is_m
        select_list = ((select_seed + mu_val) % len(channel_scale))
        iter += 1
        for can in select_list:
            t_can = tuple(can[:-1])
            flops = calculate_flops(can[:-1], model)
            if t_can in untest_dict.keys() or t_can in test_dict.keys() or flops > max_FLOPs or flops < min_FLOPs:
                continue
            can[-1] = flops
            res.append(can)
            untest_dict[t_can] = flops
            if len(res) == mutation_num:
                break

    print('mutation_num = {}'.format(len(res)), flush=True)
    return res


# crossover operation in evolution algorithm
def get_crossover(keep_top_k, layer_num, crossover_num, test_dict, untest_dict, model,max_FLOPs,min_FLOPs):
    print('crossover ......', flush=True)
    res = []
    k = len(keep_top_k)
    iter = 0
    max_iters = 10 * crossover_num
    while len(res) < crossover_num and iter < max_iters:
        id1, id2 = np.random.choice(k, 2, replace=False)
        p1 = keep_top_k[id1]
        p2 = keep_top_k[id2]
        mask = np.random.randint(low=0, high=2, size=(layer_num + 1)).astype(np.float32)
        can = p1 * mask + p2 * (1.0 - mask)
        iter += 1
        t_can = tuple(can[:-1])
        flops = calculate_flops(can[:-1], model)
        if t_can in untest_dict.keys() or t_can in test_dict.keys() or flops > max_FLOPs or flops < min_FLOPs:
            continue
        can[-1] = flops
        res.append(can)
        untest_dict[t_can] = -1
        if len(res) == crossover_num:
            break
    print('crossover_num = {}'.format(len(res)), flush=True)
    return res


def calculate_flops(can, model):
    i = 0
    width_list = [0.25, 0.5, 0.75, 1]
    for n, m in model.named_modules():
        width_id = can[i]
        # 稀疏率近似
        if isinstance(m, nn.Conv2d):
            setattr(m, 'width_mult', width_list[width_id])
            last_id = width_id  
        elif isinstance(m, nn.BatchNorm2d):
            setattr(m, 'width_id', last_id)
    input = torch.randn(1, 3, 224, 224)
    flops = profile(model, inputs=(input,))
    return flops


# random operation in evolution algorithm
def random_can(num, layer_num, test_dict, untest_dict, model,channel_scale,max_FLOPs,min_FLOPs):
    print('random select ........', flush=True)
    candidates = []
    while (len(candidates)) < num:
        import pdb; pdb.set_trace()
       # can = np.random.rand(low=0.2, high=1, layer_num).astype(np.float32)
        can = np.random.rand(layer_num+1)
        t_can = tuple(can[:-1])
        flops = calculate_flops(can[:-1], model)
        print(flops, flush=True)
        if t_can in test_dict.keys() or t_can in untest_dict.keys() or flops > max_FLOPs or flops < min_FLOPs:
            continue
        can[-1] = flops
        candidates.append(can)
        untest_dict[t_can] = -1
    print('random_num = {}'.format(len(candidates)), flush=True)
    return candidates


# select topk
def select(candidates, keep_top_k, select_num):
    print('select ......', flush=True)
    res = []
    keep_top_k.extend(candidates)
    keep_top_k = sorted(keep_top_k, key=lambda can: can[-1])
    return keep_top_k[:select_num]


def search(model, criterion, layer_num,args,save_dict):
    cnt = 1
    select_num = 50
    population_num = 50
    mutation_num = 25
    m_prob = 0.1
    crossover_num = 25
    random_num = population_num - mutation_num - crossover_num

    test_dict = {}
    untest_dict = {}
    keep_top_k = []
    keep_top_50 = []
    print(
        'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_iters = {}'.format(
            population_num, select_num, mutation_num, crossover_num, random_num, args.max_iters))

    # first 50 candidates are generated randomly
    candidates = random_can(population_num, layer_num, test_dict, untest_dict, model,channel_scale,max_FLOPs,min_FLOPs)

    start_iter = 0
    filename = './searching_snapshot.pkl'
    if os.path.exists(filename):
        data = pickle.load(open(filename, 'rb'))
        candidates = data['candidates']
        keep_top_k = data['keep_top_k']
        keep_top_50 = data['keep_top_50']
        start_iter = data['iter'] + 1

    for iter in range(start_iter, args.max_iters):

        candidates, cnt = test_candidates_model(model, criterion, candidates, cnt, test_dict)
        keep_top_50 = select(candidates, keep_top_50, select_num)
        keep_top_k = keep_top_50[0:10]

        print('iter = {} : top {} result'.format(iter, select_num), flush=True)
        for i in range(select_num):
            res = keep_top_50[i]
            print('No.{} {} Top-1 err = {}'.format(i + 1, res[:-1], res[-1]))

        untest_dict = {}
        mutation = get_mutation(keep_top_k, layer_num, mutation_num, m_prob, test_dict, untest_dict, model)
        crossover = get_crossover(keep_top_k, layer_num, crossover_num, test_dict, untest_dict, model)
        random_num = population_num - len(mutation) - len(crossover)
        rand = random_can(random_num, layer_num, test_dict, untest_dict, model)

        candidates = []
        candidates.extend(mutation)
        candidates.extend(crossover)
        candidates.extend(rand)

        print('saving tested_dict ........', flush=True)
        f = open(args.save_dict_name, 'wb')
        pickle.dump(save_dict, f)
        f.close()

        snap = {'candidates': candidates, 'keep_top_k': keep_top_k, 'keep_top_50': keep_top_50, 'iter': iter}
        pickle.dump(snap, open(filename, 'wb'))

    print(keep_top_k)
    print('finish!')


def run(args,layer_num):
    t = time.time()
    print('net_cache : ', args.net_cache)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # model = ResNet50()
    model = models.VGG('VGG11')
    model = nn.DataParallel(model.cuda())

    if os.path.exists(args.net_cache):
        print('loading checkpoint {} ..........'.format(args.net_cache))
        checkpoint = torch.load(args.net_cache)
        model.load_state_dict(checkpoint['net'])
        # print("loaded checkpoint {} epoch = {}" .format(args.net_cache, checkpoint['epoch']))

    else:
        print('can not find {} '.format(args.net_cache))
        return

    #layer_num = len(layer_num) + sum(layer_num)
    search(model, criterion, layer_num, args,save_dict)

    total_searching_time = time.time() - t
    print('total searching time = {:.2f} hours'.format(total_searching_time / 3600), flush=True)


if __name__ == '__main__':
    run(args, layer_num)
