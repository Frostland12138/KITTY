import torch
import torch.nn as nn
import math
import time
import torch.nn.functional as F
import pandas
import numpy as np

class InstanceLoss_with_curv(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss_with_curv, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.eps = 1e-8
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j,z_k,z_l):
        vec1=z_i-z_j
        vec2=z_i-z_k
        vec3=z_i-z_l
        cos_sim12 = F.cosine_similarity(vec1, vec2)
        cos_sim13 = F.cosine_similarity(vec1, vec3)
        cos_sim23 = F.cosine_similarity(vec2, vec3)
        curv= 2*np.pi-torch.acos(cos_sim12)-torch.acos(cos_sim13)-torch.acos(cos_sim23)

        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z=F.normalize(z,dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        loss += curv.mean()
        loss -= curv.mean()
        return loss,curv
