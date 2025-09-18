import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.classes_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.lineareva_fc = nn.Linear(self.resnet.rep_dim, self.classes_num)

    def forward(self, x_i, x_j, mode=1):
        hh_i = self.resnet(x_i)
        hh_j = self.resnet(x_j)

        h_i = normalize(self.instance_projector(hh_i), dim=1)
        h_j = normalize(self.instance_projector(hh_j), dim=1)


        if mode == 1:

            return self.instance_projector(hh_i), self.instance_projector(hh_j),
        elif mode == 2:
            return h_i, h_j
        elif mode == 3:
            return self.instance_projector(hh_i), self.instance_projector(hh_j)

    def forward_embedding(self, x):
        return self.resnet(x)

    def forward_linearevaluton_fc(self, x):
        return self.lineareva_fc(x)


