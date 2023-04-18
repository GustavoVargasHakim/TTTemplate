import torch

def save_checkpoint(state, mode, args, **kwargs):
    if mode == 'source':
        root = args.save + args.dataset + '_source.pth'
    elif mode == 'joint':
        info = ''
        for key in kwargs:
            info += '_' + key + str(kwargs[key])
        root = args.save + args.dataset + '_joint' + info + '.pth'
    torch.save(state, root)

def message(case, checkpoint=None, **kwargs):
    if case == 'model':
        print('Making model: {}'.format(kwargs['model']))
    elif case == 'process':
        print('Initializing Process Group...')
    elif case == 'data':
        print('Preparing dataloader for: {}'.format(kwargs['dataset']))
    elif case == 'checkpoint':
        print("Loaded checkpoint (from epoch {})".format(checkpoint['epoch']))
    elif case == 'metrics':
        print('\t\tTrain Loss \t\t Train Accuracy (%) \t\t Train time (s) \t\t Val Loss \t\t Val Acccuracy (%) \t\t Val time (s)')
    elif case == 'epoch':
        loss_train = kwargs['loss_train']
        acc_train = kwargs['acc_train']
        time_train = kwargs['time_train']
        loss_val = kwargs['loss_val']
        acc_val = kwargs['acc_val']
        time_val = kwargs['time_val']
        epoch = kwargs['epoch']
        epochs = kwargs['epochs']
        print(('Epoch %d/%d:' % (epoch, epochs)).ljust(24) +
               '%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (loss_train,acc_train,time_train,loss_val,acc_val,time_val))