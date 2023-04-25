import os
import copy
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import configuration
from dataset import prepare_dataset
from utils import utils, model_utils, test_utils

def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Loading model_____________________________________________________________________________________________________
    utils.message('model', rank = 0, model=args.model)
    path = os.path.join(args.root, 'weights', args.dataset + '_source.pth' if args.source else args.dataset + '_CUSTOM_NAME_DEPENDING_ON_METHOD.pth')
    checkpoint = torch.load(path)
    weights = checkpoint['state_dict']
    model = model_utils.create_model(args, augment=args.use_ttt, weights=weights).to(device)
    state = copy.deepcopy(model.state_dict())

    # Loading dataset___________________________________________________________________________________________________
    parameters = test_utils.get_parameters(model, mode=args.mode)
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.plr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.plr)

    # Loss Function_____________________________________________________________________________________________________
    criterion = test_utils.CustomLoss()

    # Loading dataset___________________________________________________________________________________________________
    if args.dataset == 'visda' and args.domain == 'val':
        test_loader, _ = prepare_dataset.prepare_val_data(args)
    else:
        test_loader, _ = prepare_dataset.prepare_test_data(args)

    # Test-Time Adaptation______________________________________________________________________________________________
    print('Test-Time Adaptation')
    iterations = (1,2,10,15,20,50)
    adapt_results = test_utils.AdaptMeter(length=len(test_loader.dataset), iterations=iterations)
    for batch_idx, (inputs, labels) in tqdm(enumerate(test_loader)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        model.load_state_dict(state, strict=False)
        correctness = test_utils.test_batch(model, inputs, labels)

        for i in range(1, args.niter):
            test_utils.adapt_batch(model, inputs, criterion, optimizer)
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
        args.root = '/export/livia/home/vision/gvargas/NAME_OF_PROJECT/'

    experiment(args)