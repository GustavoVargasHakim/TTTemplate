import timm
import types
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
    if augment:
        model = augment_model(model, dataset=args.dataset, **kwargs)
    if args.dataset in ['cifar10', 'cifar100']:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        model.forward = func_type(forward_small, model)
    elif args.dataset in ['visda', 'office']:
        model.forward = func_type(forward_large, model)

    #Load imagenet-pretrained weights if available
    if weights is not None:
        model.load_state_dict(weights, strict=False)

    return model

#Modify this function to add additional components/blocks to the model based on your TTT needs
def augment_model(model, dataset, **recipe):
    '''
    Augmenting model with additional modules (e.g., Y-shaped architectures, projectors, etc.)
    :param model: original timm model
    :param dataset: dataset name
    :param recipe: auxiliar information to add the new components (e.g. layer indices)
    :return: augmented model
    '''
    #Example
    layers = recipe['layers']
    if layers[1]:
        model.projector1 = nn.Conv2d(128, 128, 1)

    return model

#Modify this forward pass function to fit your TTT needs
def forward_small(self, x, feature=False):
    features = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.layer1(x)
    if hasattr(self, 'projector1'):
        proj1 = self.projector1(x)
    features.append(x)
    x = self.layer2(x)
    features.append(x)
    x = self.layer3(x)
    features.append(x)
    x = self.layer4(x)
    features.append(x)
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
    x = self.layer2(x)
    features.append(x)
    x = self.layer3(x)
    features.append(x)
    x = self.layer4(x)
    features.append(x)
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
