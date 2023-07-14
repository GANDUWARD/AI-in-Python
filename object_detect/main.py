import torchvision
# 没有发现FasterRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
import torchvision.transforms as T
import torch
import torch.utils as utils
import PennFudanDataset
from engine import train_one_epoch,evaluate
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # 在COCO上加载经过训练的预训练模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)

    #获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    #用新的头部替换预先训练好的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    #现在获取掩膜分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    #并用新的掩膜预测器替换掩膜预测器
    model.roi_heads.mask_predictor =  MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model
def main():
    #在GPU上训练，若无GPU，可选择在CPU上训练
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #我们的数据集只有两个类-背景和人
    num_classes = 2
    #使用我们的数据集和定义的转换
    dataset = PennFudanDataset.PennFudanDataset('PennFudanPed',get_transform(train=True))
    dataset_test = PennFudanDataset.PennFudanDataset('PennFudanPed',get_transform(train=False))
    #在训练和测试集中拆分数据集
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset,indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test,indices[-50:])
    #定义训练和验证数据加载器
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,num_workers=4,collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle = False,num_workers=4,collate_fn=utils.collate_fn)
    #使用我们的辅助函数获取模型
    model = get_model_instance_segmentation(num_classes)
    #将我们的模型迁移到合适的设备
    model.to(device)
    #构造一个优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)
    #和学习率调度程序
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    #训练10个epochs
    num_epochs =10
    for epoch in range(num_epochs):
        #训练一个epoch，每10次迭代打印一次
        train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq =10)
        #更新学习速率
        lr_scheduler.step()
        #在测试集上评价
        evaluate(model,data_loader_test,device=device)
    print("That is it!")

# 在COCO上加载经过训练的预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# replace the classifier with a new one
# 将分类器替换为具有用户定义的num_classe的新分类器
num_classes = 2  # 1 class (person) + background
# 获取分类器的输入参数的数量
in_features = model.roi_heads.box_predictor.cls_score.in_featrures
# 用新的头部替换预先训练好的头部
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 加载预先训练的模型进行分类和返回
# 只有功能

backbone = torchvision.models.mobilenet_v2(pretrained=True).features

# FasterRCNN需要知道骨干网中的输出通道数量。对于mobilenet_v2,它是1280，所以我们需要在这里添加它
backbone.out_channels = 1280

# 让RPN在每个空间位置生成5*3个锚点
# 具有5种不同的大小和3种不同的宽高比
# 存在一个元组【元组【int】】
# 每个特征映射可能具有不同的大小和宽高比
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# 定义一下我们将用于执行感兴趣区域裁剪的特征映射，以及重新缩放后裁剪的大小
# 如果您的主干返回Tensor，则featmap_names应为【0】
# 更一般地，主干应该返回OrderedDict【Tensor】
# 并且在featmap_names中，您可以选择要使用的功能映射。

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

# 将这些pieces放在FasterRCNN模型中
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
main()