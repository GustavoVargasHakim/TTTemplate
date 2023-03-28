import torch

from utils import model_utils
import configuration

def experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layers = tuple(map(bool, args.layers))
    model = model_utils.create_model(args, augment=True, layers=layers).to(device)

    x = torch.rand(1,3,224,224).to(device)
    model(x)
    print('Forward pass successful')

if __name__ == '__main__':
    args = configuration.argparser()
    if args.livia:
        args.dataroot = '/export/livia/home/vision/gvargas/data/'
        args.root = '/export/livia/home/vision/gvargas/MaskUp'

    experiment(args)