import copy
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
from dataset import prepare_dataset
from utils import utils, model_utils, test_utils

classes = {'cifar10': 10, 'cifar100': 100, 'visda': 12, 'office': 65}

def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    layers = tuple(map(bool, args.layers))

    '''Loading model'''
    utils.message('model', model=args.model)
    model = model_utils.create_model(args, augment=True, layers=layers).to(device)
    if args.source:
        path = args.root + '/weights/' + args.dataset + '_source.pth'
    else:
        path = args.root + '/weights/visda.pth'
    checkpoint = torch.load(path)
    weights = checkpoint['state_dict']
    model.load_state_dict(weights, strict=False)
    state = copy.deepcopy(model.state_dict())

    '''Getting parameters'''
    parameters = test_utils.get_parameters(model, layers=layers, mode=args.mode)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.plr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.plr)

    '''Loss Function'''
    criterion = test_utils.CustomLoss()

    '''Loading dataset'''
    if args.dataset in ['cifar10', 'cifar100', 'office']:
        teloader, _ = prepare_dataset.prepare_test_data(args)
    elif args.dataset == 'visda':
        if args.domain == 'val':
            teloader, _ = prepare_dataset.prepare_val_data(args)
        else:
            teloader, _ = prepare_dataset.prepare_test_data(args)

    '''Test-Time Adaptation'''
    print('Test-Time Adaptation')
    iterations = (1,2,10,15,20)
    adapt_results = test_utils.AdaptMeter(length=len(teloader.dataset), iterations=iterations)
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        model.load_state_dict(state, strict=False)
        correctness = test_utils.test_batch(model, inputs, labels)

        for i in range(1, args.niter+1):
            model.train()
            test_utils.adapt_batch(model, inputs, criterion, optimizer, K=classes[args.dataset])
            if i in iterations:
                correctness_new = test_utils.test_batch(model, inputs, labels)
                adapt_results.update(before=correctness, after=correctness_new, iter=i)

    print('--------------------RESULTS----------------------')
    print('Dataset: ', args.dataset)
    if args.dataset in ['cifar10', 'cifar100']:
        print('Perturbation: ', args.corruption)
    for iter in iterations:
        if iter <= args.niter:
            accuracy = adapt_results.accuracy(iter)
            print('Iterations: ', iter)
            print('Accuracy: ', accuracy)
            adapt_results.print_result(iter)


if __name__ == '__main__':
    args = configuration.argparser()
    if args.livia:
        args.dataroot = '/export/livia/home/vision/gvargas/data/'
        args.root = '/export/livia/home/vision/gvargas/MaskUp'

    experiment(args)