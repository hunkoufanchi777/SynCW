import torch.nn as nn
import torch.nn.functional as F


class Conv3(nn.Module):
    def __init__(self, c=1):
        super(Conv3, self).__init__()
        self.conv1 = nn.Conv2d(c, 6, 5)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.conv3 = nn.Conv2d(10, 10, 3)
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))  # 1*28*28 => 6*24*24
        out = F.max_pool2d(out, 2)  # 6*24*24 => 6*12*12
        out = F.relu(self.conv2(out))  # 6*12*12 => 10*8*8
        out = F.max_pool2d(out, 2)  # 10*8*8 => 10*4*4
        out = F.relu(self.conv3(out))  # 10*4*4 => 10*2*2
        out = F.max_pool2d(out, 2)  # 10*2*2 => 10*1*1
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def conv3(dataset):
    input_ch = 1 if 'mnist' in dataset else 3

    return Conv3(input_ch)