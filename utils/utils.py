import torch

def save_checkpoint(state, args):
    root = args.save + args.dataset + '_source.pth'
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
    elif case == 'Metrics':
        print('\t\tTrain Loss \t\t Train Accuracy \t\t Train time \t\t Val Loss  \t\t Val Acccuracy \t\t Val time')
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
    elif case == 'start_source':
        print('Starting source training')