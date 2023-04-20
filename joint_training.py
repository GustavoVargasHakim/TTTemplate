import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import configuration
from dataset import prepare_dataset
from utils import utils, dist_utils, model_utils, train_utils

def main(args):
    cudnn.benchmark = True

    # Initializing Distributed process (optional)
    if args.distributed:
        rank, current_device = dist_utils.dist_configuration(args)
    else:
        current_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        rank = 0

    # Creating Model
    if args.dataset in ['visda', 'office', 'imagenet']:
        if args.pretraining:
            weights = torch.load(args.root +'weights/NAME_OF_PRETRAINED_WEIGHTS.pth')
        else:
            weights = torch.load(args.root + 'weights/resnet50_imagenet.pth')
            if args.dataset != 'imagenet':
                del weights['fc.weight']
                del weights['fc.bias']
    else:
        if args.pretraining:
            weights = torch.load(args.root +'weights/NAME_OF_PRETRAINED_WEIGHTS.pth')
        else:
            weights = None
    model = model_utils.create_model(args, augment=True, weights=weights).to(current_device)
    if args.distributed:
        model = DDP(model, device_ids=[current_device], find_unused_parameters=True)
    utils.message('model', rank, model=args.model)

    # Model parameters and optimizer
    if args.parameters == 'special':
        parameters = train_utils.get_parameters(model, mode='splits', layers=args.layers)
    else:
        parameters = model.parameters()
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(parameters, args.lr)
    if args.dataset in ['cifar10', 'cifar100']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(150, 250), gamma=0.1)
    else:
        scheduler = None

    # Loading checkpoint
    if args.resume:
        checkpoint = torch.load(args.root + 'weights/INSERT_NAME_OF_FILE.pth')
        if args.distributed:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.dataset in ['cifar10', 'cifar100']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['start_epoch']
        utils.message('checkpoint', rank, epoch=checkpoint['start_epoch'])

    # Generating dataloader
    utils.message('data', rank, dataset=args.dataset)
    train_loader, train_sampler, val_loader, val_sampler = prepare_dataset.prepare_train_data(args)

    # Loss function
    crossentropy = torch.nn.CrossEntropyLoss()
    criterion = train_utils.CustomLoss(crossentropy=crossentropy)

    # Starting joint training
    utils.message('metrics', rank)
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # Training step
        acc_train, loss_train, tr_epoch_time = train_utils.train(model, current_device, criterion, optimizer, train_loader, augment=False, custom_forward=True)

        # Valuation step
        acc_val, loss_val, val_epoch_time = train_utils.validate(model, current_device, criterion, val_loader, augment=False, custom_forward=True)

        utils.message('epoch', rank, epoch=epoch, epochs=args.epochs, loss_train=loss_train, acc_train=acc_train, time_train=tr_epoch_time,
                                                   loss_val=loss_val, acc_val=acc_val, time_val=val_epoch_time)

        # Saving checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            model_state = model.module.state_dict() if args.distributed else model.state_dict()
            scheduler_state = scheduler.state_dict() if scheduler is not None else None
            state = {'epoch': epoch + 1,
                     'state_dict': model_state,
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler_state}
            utils.save_checkpoint(state, rank, 'joint', args)
M
if __name__=='__main__':
    args = configuration.argparser()
    main(args)