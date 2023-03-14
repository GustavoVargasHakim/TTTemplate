import argparse

def argparser():
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--root', type=str, default='/home/vhakim/projects/rrg-ebrahimi/vhakim/ClusTTT', help='Base path')
    parser.add_argument('--dataroot', type=str, default='work/')
    parser.add_argument('--save', type=str, default='work/', help='Path for base training weights')
    parser.add_argument('--livia', action='store_true', help='To use LIVIA servers directories')

    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Dataset
    parser.add_argument('--dataset', type=str, default='office', choices=('cifar10', 'cifar100', 'visda', 'office'))
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')
    '''For OfficeHome only'''
    parser.add_argument('--category', type=str, default='Real World', help='Domain category (OfficeHome)', choices=('Art', 'Clipart', 'Product', 'Real World'))

    # Model
    parser.add_argument('--model', type=str, default='resnet50', choices=('resnet50', 'resnet18', 'resnet10t'))

    # Source training
    parser.add_argument('--epochs', type=int, default=350, help='Number of base training epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Manual epoch number for restarts')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for base training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to use', choices=('sgd', 'adam'))
    parser.add_argument('--evaluate', action='store_true', help='Evaluating on evaluation set')
    parser.add_argument('--resume', type=bool, default=False, help='Continuing from checkpoint')
    parser.add_argument('--ignore-batch', action='store_true', help='Ignoring batch statistics during training/test-time')

    # Test-Time Adaptation
    parser.add_argument('--adapt', action='store_true', help='To adapt or not')
    parser.add_argument('--source', action='store_true', help='To use source model (no extra components)')
    parser.add_argument('--mode', default='layers', type=str, help='The parameters to adapt at test-time (method dependent)')
    parser.add_argument('--niter', default=10, type=int, help='Iterations for adaptation')
    parser.add_argument('--plr', type=float, default=0.001, help='Learning rate for projector training')
    '''For CIFAR-10/100-C'''
    parser.add_argument('--level', default=5, type=int, help='Level of corruption (CIFAR-10/100-C)')
    parser.add_argument('--corruption', default='gaussian_noise', help='Target dataset')
    '''For VisDA-C'''
    parser.add_argument('--domain', type=str, default='val', help='Domain dataset from VisDA-C')

    # Distributed
    parser.add_argument('--distributed', action='store_true', help='Activate distributed training')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:3456', help='url for distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
    parser.add_argument('--world-size', type=int, default=1, help='Number of nodes for training')


    return parser.parse_args()