import os
import torch
import torch.distributed as dist

def dist_configuration(args):
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
    current_device = local_rank
    torch.cuda.set_device(current_device)
    if rank == 0:
        print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    return rank, current_device

def dist_message(case, rank, checkpoint=None, **kwargs):
    if rank == 0:
        if case == 'model':
            print('From Rank: {}, ==> Making model: {}'.format(rank, kwargs['model']))
        elif case == 'process':
            print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
        elif case == 'data':
            print('From Rank: {}, ==> Preparing dataloader for: {}'.format(rank, kwargs['dataset']))
        elif case == 'checkpoint':
            print("From Rank: {}, => loaded checkpoint (from epoch {})".format(rank, checkpoint['epoch']))
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
            print('From Rank: {}, ==> Starting source training'.format(rank))
