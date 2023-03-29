import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

import configuration
from dataset import prepare_dataset
from utils import utils, dist_utils, model_utils, train_utils

def main(args):
    cudnn.benchmark = True

    '''Initializing Distributed process (optional)'''
    if args.distributed:
        rank, current_device = dist_utils.dist_configuration(args)

    '''Creating model'''
    if args.dataset in ['visda', 'office']:
        if args.pretraining:
            weights = torch.load(args.root +'/weights/NAME_OF_PRETRAINED_WEIGHTS.pth')
        else:
            weights = torch.load(args.root + '/weights/resnet50_imagenet.pth')
    else:
        if args.pretraining:
            weights = torch.load(args.root +'/weights/NAME_OF_PRETRAINED_WEIGHTS.pth')
        else:
            weights = None
    model = model_utils.create_model(args, weights=weights)
    if args.distributed:
        dist_utils.dist_message('model', rank, model=args.model)
        model = DDP(model, device_ids=[current_device], find_unused_parameters=True)
    else:
        utils.message('model', model=args.model)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if args.dataset in ['cifar10', 'cifar100']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(150, 250), gamma=0.1)

    '''Loading checkpoint'''
    if args.resume:
        checkpoint = torch.load(args.root + '/weights/INSERT_NAME_OF_FILE.pth')
        if args.distributed:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.dataset in ['cifar10', 'cifar100']:
            scheduler.load_state_dict(checkpoint['scheduler'])
        args.start_epoch = checkpoint['start_epoch']
        if args.distributed:
            dist_utils.dist_message('checkpoint', epoch=checkpoint['start_epoch'])
        else:
            utils.message('checkpoint', epoch=checkpoint['start_epoch'])

    '''Generating dataloader'''
    if args.distributed:
        dist_utils.dist_message('data', rank, data=args.dataset)
    trloader, tr_sampler, teloader, te_sampler = prepare_dataset.prepare_train_data(args)

    '''Loss function'''
    criterion = train_utils.CustomLoss()
    #crossentropy = nn.CrossEntropyLoss()

    '''Starting joint training'''
    dist_utils.dist_message('metrics', rank)
    for epoch in range(args.start_epoch, args.epochs):
        tr_sampler.set_epoch(epoch)
        te_sampler.set_epoch(epoch)

        # Training step
        acc_train, loss_train, tr_epoch_time = train_utils.train(model, criterion, optimizer, trloader, augment=False, custom_forward=True)

        # Valuation step
        acc_val, loss_val, val_epoch_time = train_utils.validate(model, criterion, trloader, augment=False, custom_forward=True)

        if args.distributed:
            dist_utils.dist_message('epoch', rank, loss_train=loss_train, acc_train=acc_train, time_train=tr_epoch_time,
                                                   loss_val=loss_val, acc_val=acc_val, time_val=val_epoch_time)
        else:
            utils.message('epoch', rank, loss_train=loss_train, acc_train=acc_train, time_train=tr_epoch_time,
                                                   loss_val=loss_val, acc_val=acc_val, time_val=val_epoch_time)

        # Saving checkpoint
        if (epoch + 1) % args.save_epochs == 0:
            model_state = model.module.state_dict() if args.distributed else model.state_dict()
            scheduler_state = scheduler.state_dict() if scheduler is not None else None
            state = {'epoch': epoch + 1,
                     'state_dict': model_state,
                     'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler_state}
            if args.distributed and rank == 0:
                utils.save_checkpoint(state, args)
            else:
                utils.save_checkpoint(state, args)


if __name__=='__main__':
    args = configuration.argparser()
    main(args)