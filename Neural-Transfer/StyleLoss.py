import torch.nn as nn
import main
import torch.nn.functional as F
class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target = main.gram_matrix(target_feature).detach()
    def forward(self,input):
        G=main.gram_matrix(input)
        self.loss = F.mse_loss(G,self.target)
        return input