import math
import torch
import torch.nn as nn
from torchvision import models


class GazeNet(nn.Module):
    def __init__(self, pretrained=False):
        super(GazeNet, self).__init__()
        self.pretrained = pretrained
        self.head_backbone = models.resnet18(pretrained=self.pretrained)
        
        self.head_backbone.fc = nn.Sequential(
            nn.Linear(512, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 256)
        )
        
        #self.head_backbone.fc = nn.Linear(1000, 256)
        self.lstm = nn.LSTM(256, 256, bidirectional=True,
                            num_layers=2, batch_first=True)
        self.last_layer = nn.Linear(512, 3)

    def forward(self, x):
        head = x
        #####################################################
        
        head_out = self.head_backbone(head).view(-1, 1, 256)
        self.lstm.flatten_parameters()
        head_out, _ = self.lstm(head_out)
        head_out = head_out.view(-1, 512)
        head_out = self.last_layer(head_out)

        """
        base_out = self.head_backbone(head.view((-1, 3) + head.size()[-2:]))

        base_out = base_out.view(head.size(0),7, 256)

        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        head_out = self.last_layer(lstm_out).view(-1,3)
        """
        #####################################################
        angular_out = head_out[:, :2]
        angular_out[:, 0:1] = math.pi*nn.Tanh()(angular_out[:, 0:1])
        angular_out[:, 1:2] = (math.pi/2)*nn.Tanh()(angular_out[:, 1:2])

        var = math.pi*nn.Sigmoid()(head_out[:, 2:3])
        var = var.view(-1, 1).expand(var.size(0), 2)
        return angular_out, var


class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1-self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o-(output_o-var_o)
        q_90 = target_o-(output_o+var_o)

        loss_10 = torch.max(self.q1*q_10, (self.q1-1)*q_10)
        loss_90 = torch.max(self.q9*q_90, (self.q9-1)*q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10+loss_90

