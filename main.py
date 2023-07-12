#训练一个模型来分类蚂蚁ants和蜜蜂bees。ants和bees各有约120张训练图 片。
# 每个类有75张验证图片。从零开始在 如此小的数据集上进行训练通常是很难泛化的。
# 由于我 们使用迁移学习，模型的泛化能力会相当好。
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import  torchvision
from torchvision import datasets ,models,transforms
import  matplotlib.pyplot as plt
import time
import os
import copy
def imshow(inp,title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std* inp + mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)    #暂停一下以便图表被更新
#训练模型
def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        #每个epoch都有一个训练和验证阶段
        for phase in ['train','val']:
            if phase =='train':
                scheduler.step()
                model.train() #Set model to training mode
            else:
                model.eval() #Set model to evaluate mode
            running_loss =0.0
            running_corrects = 0
            #迭代数据
            for inputs ,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                #零参数梯度
                optimizer.zero_grad()
                #forward
                # track history if only in train
                with torch.set_grad_enabled(phase =='train'):
                    outputs = model(inputs)
                    _,preds =torch.max(outputs,1)
                    loss = criterion(outputs,labels)

                    #backward only during the training
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                # summary
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            epoch_loss =running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]
            print('{}Loss:{:.4f}Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))

            #深度复制mo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print()
        time_elapsed = time.time()-since
        print('Training complete in {:.0f}m{:.0f}s'.format(time_elapsed//60,time_elapsed % 60))
        print('Best val Acc:{:.4f}'.format(best_acc))
        #加载最佳模型权重
        model.load_state_dict(best_model_wts)
        return model
#一个通用的展示少量预测图片的函数
def visualize_model(model,num_images=6):
    was_training = model.training
    model.eval()
    images_so_far =0
    fig =plt.figure()
    with torch.no_grad():
        for i,(inputs,labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _,preds = torch.max(outputs,1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2,2,images_so_far)
                ax.axis('off')
                ax.set_title('predicted:{}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
if __name__=='__main__':
    plt.ion()
    # 加载数据
    # 训练集数据扩充和归一化
    # 在验证集上仅需要归一化
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = './hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x])
                      for x in ['train','val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers=4)
                   for x in ['train','val']}
    dataset_sizes = {x: len(image_datasets[x])for x in ['train','val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #可视化部分图像数据
    #获取一批训练数据
    inputs,classes = next(iter(dataloaders['train']))
    #批量制作网格
    out = torchvision.utils.make_grid(inputs)
    imshow(out,title=[class_names[x] for x in classes])
    #加载预训练模型并重置最终完全连接的图层

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    #观察所有参数正在优化
    optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001,momentum=0.9)

#每7个epochs衰减LR通过设置gamma=0.1
    exp_lr_sheduler = lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
    model_ft = train_model(model_ft,criterion,optimizer_ft,exp_lr_sheduler,num_epochs=25)
    visualize_model((model_ft))
    #ConvNet作为固定特征提取器
    model_conv = torchvision.models.resnet18(pretrained = True)
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs,2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model_conv.fc.parameters(),lr=0.001,momentum=0.0)
    exp_lr_sheduler = lr_scheduler.StepLR(optimizer_conv,step_size=7,gamma=0.1)
    model_conv = train_model(model_conv,criterion,optimizer_conv,exp_lr_sheduler,num_epochs=25)
    #模型评估效果可视化
    visualize_model(model_conv)
    plt.ioff()
    plt.show()

