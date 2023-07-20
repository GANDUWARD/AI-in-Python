import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self,target,):
        super(ContentLoss,self).__init__()
        #从用于动态计算梯度的树中“分离”目标内容；
        #这是一个声明的值，而不是变量
        #否则标准的正向方法将引发错误
        self.target = target.detach()

    def forward(self,input):
        self.loss = F.mse_loss(input,self.target)
        return input
