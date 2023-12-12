import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalinfolossNet(nn.Module):
    def __init__(self):
        super(GlobalinfolossNet, self).__init__()
        self.c1 = nn.Conv2d(256,128,kernel_size=3)
        self.c2 = nn.Conv2d(128,64,kernel_size=3)

        self.l0 = nn.Linear(128+64*28*28, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, feat_H_global, feat_L_local):
        feat_L_global = F.relu(self.c1(feat_L_local))
        feat_L_global = self.c2(feat_L_global)
        feat_L_global = feat_L_global.view(feat_H_global.shape[0], -1)

        h = torch.cat((feat_H_global, feat_L_global), dim = 1)  # -> (b, N)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        h = self.l2(h) # b*1

        return h  # b*1



class LocalinfolossNet(nn.Module):
    def __init__(self):
        super(LocalinfolossNet, self).__init__()
        self.conv1 = nn.Conv2d(256+128, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, feat_H_local, feat_L_local):
        x = torch.cat((feat_L_local, feat_H_local), dim=1)

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.conv3(h)
        return h # (4,1,32,32)



class DeepMILoss(nn.Module):
    def __init__(self,weight_miloss=100,weight_LMI=0.5,weight_GMI=0.5):
        super(DeepMILoss, self).__init__()

        self.global_d = GlobalinfolossNet()
        self.local_d = LocalinfolossNet()

        self.weight_miloss = weight_miloss
        self.weight_LMI = weight_LMI
        self.weight_GMI = weight_GMI

        self.l = nn.Linear(256*32*32, 128)



    def forward(self, feat_L_local, feat_L_local_prime, feat_H_local):
        feat_H_global = self.l(feat_H_local.view(feat_H_local.shape[0], -1))

        feat_H_global_exp = feat_H_global.unsqueeze(-1).unsqueeze(-1)
        feat_H_local = feat_H_global_exp.expand(-1, -1, 32, 32)

        Ej = -F.softplus(-self.local_d(feat_H_local, feat_L_local)).mean() # positive pairs
        Em = F.softplus(self.local_d(feat_H_local, feat_L_local_prime)).mean() # negetive pairs
        LOCAL = (Em - Ej) * self.weight_LMI

        Ej = -F.softplus(-self.global_d(feat_H_global, feat_L_local)).mean() # positive pairs
        Em = F.softplus(self.global_d(feat_H_global, feat_L_local_prime)).mean() # negetive pairs
        GLOBAL = (Em - Ej) * self.weight_GMI


        ToT = (LOCAL + GLOBAL) * self.weight_miloss


        return LOCAL, GLOBAL, ToT
