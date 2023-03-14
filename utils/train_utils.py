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

def train(model, criterion, optimizer, train_loader, augment=False, custom_forward=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    start = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

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

def validate(model, criterion, val_loader, augment=False, custom_forward=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

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
        self.crossentropy = kwargs['crossentropy']

    def forward(self, output, target, **kwargs):
        loss_ce = self.crossentropy(output, target)
        loss_ce += 0.0

        return loss_ce