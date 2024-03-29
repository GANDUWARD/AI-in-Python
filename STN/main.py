import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import Net
#可视化STN结果
def convert_image_np(inp):
    #Convert a Tensor to numpy image.
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    return inp

#我们想要在训练之后可视化空间变换器层的输出
#我们使用STN可视化一批输入图像和相应的变换批次
def visualize_stn():
    with torch.no_grad():
        #Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        trainformed_input_tensor = model.stn(data).cpu()
        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))
        out_grid =  convert_image_np(torchvision.utils.make_grid(trainformed_input_tensor))
        #Plot the results side-by-side
        f,axarr = plt.subplots(1,2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

if __name__ == '__main__':
    plt.ion() #交互模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #训练数据集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.',train=True,download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ])),batch_size = 64,shuffle = True,num_workers =4)
    #测试数据集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='.', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)
    model = Net.Net().to(device)
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    def train(epoch):
        model.train()
        for batch_idx,(data,target) in enumerate(train_loader):
            data,target = data.to(device) , target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,target)
            loss.backward()
            optimizer.step()
            if batch_idx %500 == 0:
                print('Train Epoch:{} [{}/{}({:.0f}%)]\tLoss:{:.6f}'
                      .format(epoch,batch_idx*len(data),
                    len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
    #测试程序，用于测量STN在MNIST上的性能。
    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            for data,target in test_loader:
                data,target = data.to(device),target.to(device)
                output = model(data)
                #累加批量损失
                test_loss += F.nll_loss(output,target,size_average=False).item()
            test_loss /= len(test_loader.dataset)
            print('\nTest set:Average loss:{:.4f},Accuracy: {}/{} ({:.0f}%)\n'
            .format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))
    for epoch in range(1,20+1):
        train(epoch)
        test()
    #在某些输入批处理上可视化STN转换
    visualize_stn()

    plt.ioff()
    plt.show()