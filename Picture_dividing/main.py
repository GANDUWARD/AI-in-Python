import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import Net


def imshow(img):
    img = img/2 +0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=2)
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    dataiter= iter(trainloader)
    images , labels = dataiter.__next__()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net.Net()
    for epoch in range(2):
        running_loss=0.0
        for i,data in enumerate(trainloader,0):
            inputs,labels=data

            net.optimizer.zero_grad()
            outputs = net(inputs)
            loss = net.criterion(outputs,labels)
            loss.backward()
            net.optimizer.step()

            running_loss += loss.item()
            if i%2000 == 1999:
                print('[%d,%5d] loss: %.3f' %(epoch+1,i+1,running_loss/2000))
                running_loss = 0.0
    print('Finished Training')
    outputs = net(images)
    _,predicted = torch.max(outputs,1)
    print('Pridicted: ',' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    print('Finished Pridicted')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images,labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            c = (predicted==labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label]+=c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * class_correct[i] / class_total[i]))

