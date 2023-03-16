import timm
import types
import torch
import torch.nn as nn

def create_model(args, weights=None, augment=False, **kwargs):
    '''
    Function to create (custom) models for TTT/TTA
    :param args: parsed arguments
    :param weights: weights file (.pth)
    :param kwargs: Additional arguments specific to the method
    :return: model
    '''
    func_type = types.MethodType
    classes = {'cifar10': 10, 'cifar100': 100, 'visda': 12, 'office': 65}
    model = timm.create_model(args.model, features_only=True, pretrained=False)
    model.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(2048, classes[args.dataset])
    #Augmenting model for a TTT task
    if kwargs['augment']:
        model = augment_model(model, dataset=args.dataset, layers=kwargs['layers'])
    if args.dataset in ['cifar10', 'cifar100']:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
        model.forward = func_type(forward_small, model)
    elif args.dataset in ['visda', 'office']:
        model.forward = func_type(forward_large, model)

    #Load imagenet-pretrained weights if available
    if weights is not None and args.dataset in ['visda', 'office']:
        del weights['fc.weight']
        del weights['fc.bias']
        model.load_state_dict(weights, strict=False)

    return model

#Modify this function to add additional components/blocks to the model based on your TTT needs
def augment_model(model, **recipe):
    '''
    Augmenting model with additional modules (e.g., Y-shaped architectures, projectors, etc.)
    :param model: original timm model
    :param recipe: auxiliar information to add the new components (e.g. layer indices)
    :return: augmented model
    '''
    #Example
    layers = recipe['layers']
    dataset = recipe['dataset']
    model.inference = False
    if layers[0]:
        channels, resolution = model_sizes(dataset, layer=1)
        model.mask1 = MaskAdapter(channels, resolution)
    if layers[1]:
        channels, resolution = model_sizes(dataset, layer=2)
        model.mask2 = MaskAdapter(channels, resolution)
    if layers[2]:
        channels, resolution = model_sizes(dataset, layer=3)
        model.mask3 = MaskAdapter(channels, resolution)
    if layers[3]:
        channels, resolution = model_sizes(dataset, layer=4)
        model.mask4 = MaskAdapter(channels, resolution)

    return model

#Modify this forward pass function to fit your TTT needs
def forward_small(self, x, feature=False):
    features = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.layer1(x)
    features.append(x)
    if hasattr(self, 'mask1'):
        x = self.mask1(x, inference=self.inference)
    x = self.layer2(x)
    features.append(x)
    if hasattr(self, 'mask2'):
        x = self.mask2(x, inference=self.inference)
    x = self.layer3(x)
    features.append(x)
    if hasattr(self, 'mask3'):
        x = self.mask3(x, inference=self.inference)
    x = self.layer4(x)
    features.append(x)
    if hasattr(self, 'mask4'):
        x = self.mask4(x, inference=self.inference)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    if feature:
        return x, features
    else:
        return x

#Modify this forward pass function to fit your TTT needs
def forward_large(self, x, feature=False):
    features = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    features.append(x)
    if hasattr(self, 'mask1'):
        x = self.mask1(x, inference=self.inference)
    x = self.layer2(x)
    features.append(x)
    if hasattr(self, 'mask2'):
        x = self.mask2(x, inference=self.inference)
    x = self.layer3(x)
    features.append(x)
    if hasattr(self, 'mask3'):
        x = self.mask3(x, inference=self.inference)
    x = self.layer4(x)
    features.append(x)
    if hasattr(self, 'mask4'):
        x = self.mask4(x, inference=self.inference)
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    if feature:
        return x, features
    else:
        return x

def model_sizes(dataset, layer):
    if dataset == 'cifar10' or dataset == 'cifar100':
        if layer == 0:
            channels, resolution = 64, 32
        if layer == 1:
            channels, resolution = 256, 32
        if layer == 2:
            channels, resolution = 512, 16
        if layer == 3:
            channels, resolution = 1024, 8
        if layer == 4:
            channels, resolution = 2048, 4

    elif dataset in ['visda', 'office', 'imagenet']:
        if layer == 0:
            channels, resolution = 64, 56
        if layer == 1:
            channels, resolution = 256, 56
        if layer == 2:
            channels, resolution = 512, 28
        if layer == 3:
            channels, resolution = 1024, 14
        if layer == 4:
            channels, resolution = 2048, 7
    else:
        print('Dataset not found!')
        channels, resolution = None, None

    return channels, resolution

'''----------------------ADAPTERS-------------------------'''
class Threshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m):
        return (m > 0.5).float()

    @staticmethod
    def backward(ctx, m):
        return m

class MaskAdapter2(nn.Module):
    def __init__(self, channels, size):
        super(MaskAdapter, self).__init__()
        self.mask = nn.Parameter(torch.ones(channels,size,size), requires_grad=True)
        self.τ = nn.Parameter(torch.rand(1), requires_grad=True)
        self.adapter = nn.Parameter(torch.ones(channels,size,size), requires_grad=True)

    def forward(self, x, inference=False, keep=True):
        if not inference:
            u = torch.rand(1)
            l = torch.log(u) - torch.log(1-u)
            p_mask = torch.sigmoid((self.mask + l.to(x.device))/self.τ)
        else:
            p_mask = torch.sigmoid((self.mask)/self.τ)
        b_mask = Threshold.apply(p_mask)
        x_masked = x*b_mask
        x_adapt = x_masked*self.adapter
        if keep:
            x_normal = x*torch.logical_xor(torch.ones(x.shape).to(x.device), b_mask)
            x_adapt = x_adapt + x_normal

        return x_adapt

class MaskAdapter(nn.Module):
    def __init__(self, channels, size):
        super(MaskAdapter, self).__init__()
        self.mask = nn.Parameter(torch.ones(channels,size,size), requires_grad=True)
        self.τ = nn.Parameter(torch.rand(1), requires_grad=True)
        self.adapter = nn.Parameter(torch.ones(channels,size,size), requires_grad=True)

    def forward(self, x, inference=False, keep=True):
        if not inference:
            u = torch.rand(1)
            l = torch.log(u) - torch.log(1-u)
            p_mask = torch.sigmoid((self.mask + l.to(x.device))/self.τ)
        else:
            p_mask = torch.sigmoid((self.mask)/self.τ)
        b_mask = Threshold.apply(p_mask)
        if keep:
            #x_adapt = x*b_mask*self.adapter + x*(1.0 - b_mask)
            x_adapt = torch.where(b_mask.bool(), x*self.adapter, x)
        else:
            x_adapt = x*b_mask*self.adapter

        return x_adapt