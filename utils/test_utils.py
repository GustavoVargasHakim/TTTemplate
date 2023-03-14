import torch
import torch.nn as nn

#Modify this custom function to get model's parameters based on your TTT/TTA needs
def get_parameters(model, mode='layers', **kwargs):
    '''
    Extracting parameters to adapt from a model
    :param model: joint-trained/source-trained model
    :param mode: type of extraction (e.g., 'layers' for updating layer blocks)
    :return: optimizer-ready parameters
    '''
    if mode == 'layers':
        layer = kwargs['layer']
        if layer == 1:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1])
        elif layer == 2:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2])
        elif layer == 3:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3])
        elif layer == 4:
            parameters = nn.ModuleList([model.conv1, model.bn1, nn.ReLU(inplace=True), model.layer1, model.layer2, model.layer3, model.layer4])

    return parameters.parameters()

#Modify this custom loss function for your TTA needs (no crossentropy is allowed)
class CustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, **kwargs):
        loss = 0.0

        return loss

def test_batch(model, inputs, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        correctness = predicted.eq(labels)
    return correctness

#Modify this custom function to adapt the model's parameters for one iteration based on your TTT/TTA needs
def adapt_batch(model, inputs, criterion, optimizer, **kwargs):
    output = model(inputs)
    loss = criterion(output)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()

class AdaptMeter():
    """Computes and stores the predictions of batches at different iterations of adaptation"""
    def __init__(self, length, iterations = (1, 3, 10, 15, 20, 50)):
        self.length = length
        self.good_good = {iter: [] for iter in iterations}
        self.good_bad = {iter: [] for iter in iterations}
        self.bad_good = {iter: [] for iter in iterations}
        self.bad_bad = {iter: [] for iter in iterations}

    def update(self, before, after, iter):
        for i in range(len(after.tolist())):
            if before[i] == True and after[i] == True:
                self.good_good[iter].append(1)
            elif before[i] == True and after[i] == False:
                self.good_bad[iter].append(1)
            elif before[i] == False and after[i] == True:
                self.bad_good[iter].append(1)
            elif before[i] == False and after[i] == False:
                self.bad_bad[iter].append(1)

    def accuracy(self, iter):
        return (len(self.good_good[iter]) + len(self.bad_good[iter]))/self.length

    def print_result(self, iter):
        print('Good first, good after: ', len(self.good_good[iter]))
        print('Good first, bad after: ', len(self.good_bad[iter]))
        print('Bad first, good after: ', len(self.bad_good[iter]))
        print('Bad first, bad after: ', len(self.bad_bad[iter]))


'''adapt = AdaptMeter(20, (1,3,10,15))
for i in range(4):
    y = torch.FloatTensor(5).uniform_() > 0.5
    print('Labels: ', y)
    for iter in range(1, 16):
        if iter in (1,3,10,15):
            r = torch.FloatTensor(5).uniform_() > 0.5
            print('Predictions: ', r)
            adapt.update(y, r, iter=iter)

for iter in [1,3,10,15]:
    print('Accuracy: ', adapt.accuracy(iter))
    adapt.print_result(iter)'''
