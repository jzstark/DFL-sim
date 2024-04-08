
from base import BaseAgent, Message, Channel
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
import copy
from collections import OrderedDict

device = torch.device("cpu")
batch_size = 64
test_batch_size = 1000
learning_rate = 0.0001
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_dataset_train = datasets.MNIST('./data', train=True, download=True,
                    transform=transform)
#HACK!
subset_indices = list(range(500))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleDNNAgent(BaseAgent):

    _id_counter = 0

    def __init__(self, vehID: str, chan: Channel) -> None:
        super().__init__(vehID, chan) 
        self.model = Net().to(device)
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)

        #HACK!
        #subset_indices = [400 *  SimpleDNNAgent._id_counter, 
        #                  400 * (SimpleDNNAgent._id_counter + 1) - 1]
        #dataset1 = Subset(full_dataset_train, subset_indices)
        dataset1 = full_dataset_train
        dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
        
        train_kwargs = {'batch_size': batch_size}
        test_kwargs  = {'batch_size': test_batch_size}
        cuda_kwargs  = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}

        self.train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        SimpleDNNAgent._id_counter += 1
    
    
    def get_data(self):
        return self.model.state_dict()
    

    def get_comm_data(self):
        return self.data_change
    
    # return n - m
    def _sd_diff_sd(self, oldsd, newsd) -> OrderedDict:
        # assume the two dnn are of the same network architecture
        sdd = copy.deepcopy(oldsd)
        for name in sdd.keys():
            sdd[name] = newsd[name] - oldsd[name]
        return sdd 


    def updateLocalData(self):
        old_data = self.get_data()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
        print("training for one epoch in vehicle #%s" % self.id)
        return self._sd_diff_sd(old_data, self.get_data())
    

    def aggregate(self):
        # aggregate list of parameters and update mine
        datalist = self.flat_cached_data() # state_dicts 
        if datalist == []: return
        self.test()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                layer_params = [sd[name].cpu().numpy() for sd in datalist]
                grad = np.mean(layer_params, axis=0)
                param.copy_(param + torch.tensor(grad).to(device))
                #if name == 'fc2.bias':
                #    for p in layer_params:
                #        print(p)
                #    print("fuck!")
                #    print(avg)
                #    print(self.model.state_dict()[name])
        self.test()
        return
        

    def test(self):
        #TODO: test accuracy on a non-bias validate dataset 
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        print('\nTest on vehicle {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.id, test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        return test_loss