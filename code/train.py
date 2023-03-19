import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import cv2
import os
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

os.environ["CUDA_VISIBLE_DEVICES"] = "3,7"

t1t2 = []
label = []
for i in range(10):
    data=np.load('./dataset/{}/data.npz'.format(i))
    data1 = data["ct"]
    data1 = np.transpose(data1,[0,3,1,2])
    t1t2.append(data1)
    data2 = data["label"]
    data2 = np.transpose(data2,[0,3,1,2])
    label.append(data2)

#t1t2[0].shape = (720,2,256,256)
#label[0].shape = (720,3,256,256)
#print(np.array(t1t2[0]).shape)

alldata = []
for a in range(10):
    data = np.concatenate((np.array(t1t2[a]),np.array(label[a])),axis = 1 )
    alldata.append(data)

#alldata[0] = (720,5,256,256)
#print(np.array(alldata[0]).shape)


class datasets(Dataset):
    def __init__(self, data ,count , transform=None):
        self.transform = transform
        self.img = data[0:10][count]
        self.mak = data[0:10][count]
    def __getitem__(self, index):
        input_image = self.img[index][0:2]
        input_mask = self.mak[index][2:5]
        return input_image , input_mask

    def __len__(self):
        return len(self.img)


def train(model, loss_function1,loss_function2, optimizer, epochs):
    for epoch in range(1, epochs+1):

        print("Epoch: {}/{}".format(epoch, epochs))

        # train the model #
        model.train()

        for count in range(10):
            train_set = datasets(data=alldata,count = count)
            train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=False, num_workers=0)

            a = 0
            for data, target in tqdm(train_loader):
                train_loss = 0.0
                # move tensors to GPU if CUDA is available
                data = data.type(torch.FloatTensor)/255
                target = target.type(torch.FloatTensor)/255

                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss1 = loss_function1(output, target)
                loss2 = loss_function2(output, target)
                loss = 0.7 * loss1 + 0.3 * loss2
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
                '''pred = output.data.round()
                correct += (2 * pred * target).cpu().numpy().sum((2,3))
                total += target.cpu().numpy().sum((2,3)) + pred.cpu().numpy().sum((2,3))
                train_acc += (correct/total).sum()
                a = a+1'''


            train_loss = train_loss/len(train_loader.dataset)
            '''train_acc = train_acc / a'''
            print('\tTraining Loss: {:.6f} '.format(train_loss))
            model.eval()

    torch.save(model.state_dict(), 'model_dlinknet2.pth')
    return 0

#model = DinkNet50()
model = DinkNet34()
LR = 0.001
n_epochs = 80

#optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

'''class WeightBCELoss(nn.Module):
  def __init__(self):
    super(WeightBCELoss, self).__init__()

  def forward(self,pred,label):
    b,c,h,w = pred.shape
    eps = 0.0000001
    weight = 1 - torch.sum(pred, (2, 3),keepdim=True)/(h*w)
    pred = torch.clamp(pred, eps, 1-eps)
    true_loss = weight * label * torch.log(pred)
    false_loss = (1-weight) * (1-label) * torch.log(1-pred)
    return -torch.mean(true_loss + false_loss)'''

BCE = nn.BCELoss()
#BCE# = WeightBCELoss()
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

DICE = DiceLoss()


#GPU
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.cuda()

#main
#train
train(model=model,loss_function1 = BCE , loss_function2 = DICE, optimizer=optimizer, epochs=n_epochs)


#test
'''def test(model , test_loader , loss_function ):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    device = torch.device('cuda')
    model.cuda()
    model.load_state_dict(torch.load('model_dlinknet.pth'))
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(enumerate(test_loader,1)):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            data = data/255
            target = target / 255
            output = model(data)
            loss = loss_function(output, target)
            test_loss += loss.item()*data.size(0)
            pred = output.data.round()
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.sum()

        print('Test Loss: {:.6f}'.format(test_loss))

        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))
    return 0
#test_acc
#test(model=model , test_loader=test_loader , loss_function=BCE)'''
