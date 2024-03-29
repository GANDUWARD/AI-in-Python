import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

import ContentLoss
import Normalization
import StyleLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
# 期望的深度层来计算样式\内容损失
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img,
                               content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    # 规范化模块
    normalization = Normalization.Normalization(normalization_mean, normalization_std).to(device)
    #拥有可迭代的访问权限或列出内容/系统损失
    content_losses = []
    style_losses = []
    #假设cnn是一个`nn.Sequential`
    #所以创建一个新的`nn.Sequential`来放入应该顺序激活的模块
    model = nn.Sequential(normalization)
    i = 0 #increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name = 'relu_{}'.format(i)
            #对于下面插入的ContentLoss 与 StyleLoss
            #本地版本不能很好的发挥作用。所以我们在这里替换不合适的
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name ='pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer:{}'.format(layer.__class__.__name__))
        model.add_module(name,layer)
        if name in  style_layers:
            #加入内容损失：
            target = model(content_img).detach()
            content_loss = ContentLoss.ContentLoss(target)
            model.add_module("content_loss_{}".format(i),content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            # 加入风格损失：
            target_feature = model(style_img).detach()
            style_loss = StyleLoss.StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i),style_loss)
            style_losses.append(style_loss)

    #现在在最后的内容和风格损失之后剪掉了图层
    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],ContentLoss.ContentLoss) or isinstance(model[i],StyleLoss.StyleLoss):
            break
    model = model[:(i+1)]
    return model,style_losses,content_losses

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device,torch.float)
def imshow(tensor,title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
def gram_matrix(input):
    a,b,c,d = input.size() #a=batch size(=1)
    #特征映射b=number
    #（c，d）= dimensions of a f.map(N=c*d)
    features =  input.view(a*b,c*d) #resize F_XL into \hat F_XL
    G = torch.mm(features,features.t()) #compute the gram product
    #通过除以每个特征映射中的元素数来标准化gram矩阵的值
    return G.div(a*b*c*d)
def get_input_optimizer(input_img):
    #此行显示输入是需要渐变的参数
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
def run_style_transfer(cnn,normalization_mean,normalization_std,
                       content_img,style_img,input_img,num_steps=300,style_weight=1000000,content_weight=1):
    """Run the style transfer"""
    print('Building the style transfer model..')
    model,style_losses,content_losses = get_style_model_and_losses(cnn,normalization_mean,normalization_std,style_img
                                                                   ,content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0]<= num_steps:
        def closure():
            #更正更新的输入图像的值
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_losses:
                style_score+= sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss =style_score+content_score
            loss.backward()
            run[0]+=1
            if run[0]%50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f}Content Loss: {:4f}'.format(style_score.item(),content_score.item()))
            return style_score +content_score
        optimizer.step(closure)
    input_img.data.clamp_(0,1)
    return input_img

if __name__=='__main__':
    #所需的输出图像大小
    imsize = 512 if torch.cuda.is_available() else 128 #没有gpu就用小尺寸
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    style_img = image_loader("./data/images/neural-style/picasso.jpg")
    content_img = image_loader("./data/images/neural-style/dancing.jpg")

    assert style_img.size() == content_img.size()   #需要相同尺寸的风格照片与内容照片
    #现在，创建一个方法，通过重新将图片转换成PIL格式展示，并使用Plt.imshow展示它的拷贝。尝试展示以确保正确导入
    unloader = transforms.ToPILImage()
    plt.ion()
    plt.figure()
    imshow(style_img,title='Style Image')
    plt.figure()
    imshow(content_img,title='Content Image')
    input_img = content_img.clone()
    #如果您想使用白噪声而取消注释以下行：
    #input_img = torch.randn(content_img.data.size(),device = device)
    #将原始输入图像添加到图中：
    plt.figure()
    imshow(input_img,title='Input Image')
    output= run_style_transfer(cnn,cnn_normalization_mean,cnn_normalization_std,content_img,style_img,input_img)
    plt.figure()
    imshow(output,title='Output Image')
    #sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

