import torch 
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import torchvision
from torch.autograd import Variable

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x) + F.relu(-x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        out = x
        return out
dummy_input = Variable(torch.rand(13, 1, 28, 28))

model = Net1()
out = model(dummy_input)
with SummaryWriter('log/test/log/', comment='Net1') as w:
    
    w.add_graph(model, (dummy_input, ))
