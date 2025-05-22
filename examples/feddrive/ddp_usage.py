"""run.py:"""
#!/usr/bin/env python
import torch.distributed.pipeline as pp
import torch.nn as nn
import torch
from torch.distributed.pipelining import ScheduleGPipe

#genreate a Lenet model and split it into 2 stages
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = flow.nn.Flatten()
        self.linear0 = flow.nn.Linear(28*28, 512)
        self.relu0 = flow.nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear0(out)
        out = self.relu0(out)
        return out

class Stage1Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = flow.nn.Linear(512, 512)
        self.relu1 = flow.nn.ReLU()
        self.linear2 = flow.nn.Linear(512, 10)
        self.relu2 = flow.nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        return out

class PipelineModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_stage0 = Stage0Module()
        self.m_stage1 = Stage1Module()

        self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
        self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

    def forward(self, x):
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

