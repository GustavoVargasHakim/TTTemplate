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
    utils.message('model', rank=0, model=args.model)
    model = model_utils.create_model(args).to(device)
    if args.source:
        path = args.root + 'weights/' + args.dataset + '_source.pth'
    else:
        path = args.root + 'weights/' + args.dataset + '_CUSTOM_ROOT_DEPENDING_ON_METHOD' + '.pth'
    checkpoint = torch.load(path)
    weights = checkpoint['state_dict']
    model.load_state_dict(weights)

    # Loading dataset___________________________________________________________________________________________________
    if args.dataset in ['cifar10', 'cifar100', 'office']:
        teloader, _ = prepare_dataset.prepare_test_data(args)
    elif args.dataset == 'visda':
        if args.domain == 'val':
            teloader, _ = prepare_dataset.prepare_val_data(args)
        else:
            teloader, _ = prepare_dataset.prepare_test_data(args)

    # Testing___________________________________________________________________________________________________________
    print('Test-Time Adaptation')
    correct = 0.0
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        correctness = test_utils.test_batch(model, inputs, labels)
        correct += correctness.cpu().sum()

    print('--------------------RESULTS----------------------')
    print('Dataset: ', args.dataset)
    if args.dataset in ['cifar10', 'cifar100']:
        print('Perturbation: ', args.corruption)
    print('Accuracy: ', (correct/len(teloader.dataset)).item())

if __name__ == '__main__':
    args = configuration.argparser()
    if args.livia:
        args.dataroot = '/export/livia/home/vision/gvargas/data/'
        args.root = '/export/livia/home/vision/gvargas/NAME_OF_PROJECT/'

    experiment(args)