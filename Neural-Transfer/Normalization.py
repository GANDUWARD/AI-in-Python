import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization,self).__init__()
        """
        .view the mean and std to make them[C*1*1] so that they can directly
        work with image Tensor of shape [B*C*H*W]. B is batch size .C is the number of channels
        H is height and W is width.
        """
        self.mean = torch.Tensor(mean).view(-1,1,1)
        self.std = torch.Tensor(std).view(-1,1,1)
    def forward(self,img):
        #normalize img
        return (img-self.mean)/self.std
