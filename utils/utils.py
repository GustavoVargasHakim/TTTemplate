import torch

#Returns the an integer code corresponding to the chosen layers (useful for TTT methods)
def layer_codes(layers):
    '''
    Returns integer code for
    :param layers: List of integers defining layers (e.g., [1,0,0,0] for layer 1)
    :return: integer code to add to save_checkpoint (e.g., 1)
    '''
    layer_code = ''
    for elem in layers:
        layer_code += str(elem)
    codes = {'1000':1, '0100':2, '0010':3, '0001':4,
             '1100':12, '0110':23, '0011':34, '1010':13, '0101':24, '1001':14,
             '1110':123, '0111':234, '1101':124, '1011':134, '1111':'all'}
    return codes[layer_code]

def save_checkpoint(args, state, rank, mode, **kwargs):
    if rank == 0:
        if mode == 'source':
            root = args.save + args.dataset + '_source.pth'
        elif mode == 'joint':
            info = ''
            for key in kwargs:
                info += '_' + key + str(kwargs[key])
            root = args.save + args.dataset + '_joint' + info + '.pth'
        torch.save(state, root)

def message(case, rank, checkpoint=None, **kwargs):
    if rank == 0:
        if case == 'model':
            print('Making model: {}'.format(kwargs['model']))
        elif case == 'process':
            print('Initializing Process Group...')
        elif case == 'data':
            print('Preparing dataloader for: {}'.format(kwargs['dataset']))
        elif case == 'checkpoint':
            print("Loaded checkpoint (from epoch {})".format(checkpoint['epoch']))
        elif case == 'metrics':
            print('\t\tTrain Loss\t\tTrain Acc (%)\t\t Val Loss\t\t Val Acc (%)\t\tTime (s)')
        elif case == 'epoch':
            loss_train = kwargs['loss_train']
            acc_train = kwargs['acc_train']
            loss_val = kwargs['loss_val']
            acc_val = kwargs['acc_val']
            epoch = kwargs['epoch']
            epochs = kwargs['epochs']
            time_epoch = kwargs['time_epoch']
            print(('Epoch %d/%d:' % (epoch, epochs)).ljust(24) +
                   '%.2f\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f' % (loss_train,acc_train,loss_val,acc_val,time_epoch))