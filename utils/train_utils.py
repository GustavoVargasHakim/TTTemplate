import time
import torch
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Modify this custom function to get model's parameters based on your TTT/TTA needs
def get_parameters(model, mode='layers', distributed=False, **kwargs):
    '''
    Extracting parameters to adapt from a model
    :param model: joint-trained/source-trained model
    :param mode: type of extraction (e.g., 'layers' for updating layer blocks)
    :return: optimizer-ready parameters
    '''
    if mode == 'full':
        return model.parameter()

    if mode == 'layers': #Example to train layer blocks up to specific number
        if distributed:
            model = model.module
        layer = kwargs['layer']
        if layer == 1:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1])
        elif layer == 2:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2])
        elif layer == 3:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2,
                                        model.layer3])
        elif layer == 4:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2,
                                        model.layer3, model.layer4])
        return parameters.parameters()

    if mode == 'adapters': #Example for extra modules
        if distributed:
            model = model.module
        layers = kwargs['layers']
        layers = tuple(map(bool, layers))
        parameters = []
        if layers[0]:
            parameters += list(model.adapter1.parameters())
        elif layers[1]:
            parameters += list(model.adapter2.parameters())
        elif layers[2]:
            parameters += list(model.adapter3.parameters())
        elif layers[3]:
            parameters += list(model.adapter4.parameters())

        return parameters
def train(model, device, criterion, optimizer, train_loader, augment=False, custom_forward=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    # switch to training mode
    model.train()

    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Optional data augmentation
        if augment:
            images, labels_aug = augment_func(images, labels)
        else:
            labels_aug = None

        #Compute output and loss
        if custom_forward:
            output, loss = forward_func(model, images, labels, labels_aug, criterion) #Custom function when having test-time training methods
        else:
            output = model(images)
            loss = criterion(output, labels)

        #Compute accuracy
        acc1 = accuracy(output, labels, topk=(1,))
        losses.update(loss.item(), images.size(0))
        acc.update(acc1[0], images.size(0))

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - start)

    return acc.avg, losses.avg, batch_time.avg

def validate(model, device, criterion, val_loader, augment=False, custom_forward=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Optional data augmentation
            if augment:
                images, labels_aug = augment_func(images, labels)
            else:
                labels_aug = None

            # Compute output and loss
            if custom_forward:
                output, loss = forward_func(model, images, labels, labels_aug, criterion)  # Custom function when having test-time training methods
            else:
                output = model(images)
                loss = criterion(output, labels)

            # measure accuracy and record loss
            acc1 = accuracy(output, labels, topk=(1,))
            losses.update(loss.item(), images.size(0))
            acc.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start)

    return acc.avg, losses.avg, batch_time.avg

#Modify this custom data augmentation function to your TTT needs
def augment_func(images, labels):
    '''
    Custom data augmentation function (for TTT-like methods)
    :param images: input images before augmentation
    :param labels: input labels before augmentation
    :return: the transformed images and labels. Original and transformed images should be concatenated to avoid double forward passes
    '''
    images_aug, labels_aug = images, labels
    return images_aug, labels_aug

#Modify this custom forward function to your TTT needs
def forward_func(model, images, labels, labels_aug, criterion):
    '''
    Custom forward pass for TTT methods
    :param model: custom model (must compute the normal and auxiliary tasks end to end)
    :param criterion: custom loss function (must include crossentropy and all auxiliary losses)
    :return: the prediction output and the loss function
    '''
    output = torch.zeros(10)
    loss = torch.tensor([0.0])
    return output, loss

#Modify this custom loss function for your TTT needs
class CustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__()
        self.supervised = kwargs['supervised']

    def forward(self, output, target, **kwargs):
        loss_ce = self.supervised(output, target)
        loss_ce += 0.0 #Add additional losses using kwargs (for Test-time training methods)

        return loss_ce