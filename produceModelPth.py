import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from models import wideResnet as WRN
from models import Resnet as Resnet
from models import  densenet as Densenet
from models import  BigResnet
import torch.nn.functional as F
import numpy as np
from models.autoaugment import CIFAR10Policy
from models.cutout import Cutout
import time, os
from torch.utils.data import DataLoader
import torch.nn as nn
from models import VGG  # 新增导入
# 随机种子
np.random.seed(0)
torch.manual_seed(0)
parser = argparse.ArgumentParser(description='produce model xxx.pth')
parser.add_argument('--model', default="WRN1602", type=str, help="resnet20|resnet32|resnet56|resnet18|resnet34|resnet50|WRN1602|WRN4002|densenet40|densenet100_12|densenet100_24")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100")
parser.add_argument('--epoch', default=300, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="./data", type=str)
parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--init_lr', default=0.01, type=float)
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# '0,1,2,3,4,5'

global numclass
global trainloader
global testloader


# 数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 数据加载
trainset = torchvision.datasets.CIFAR100(
    root=args.dataset_path,
    train=True,
    download=True,
    transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root=args.dataset_path,
    train=False,
    download=True,
    transform=transform_test
)
numclass = 100

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=4
)



#模型选择
global studentnet
if args.model == "resnet20":
    studentnet = Resnet.resnet20(num_classes=numclass)
elif args.model == "resnet32":
    studentnet = Resnet.resnet32(num_classes=numclass)
elif args.model == "resnet56":
    studentnet = Resnet.resnet56(num_classes=numclass)
elif args.model == "resnet18":
    studentnet = BigResnet.resnet18(num_classes=numclass)
elif args.model == "resnet34":
    studentnet = BigResnet.resnet34(num_classes=numclass)
elif args.model == "resnet50":
    studentnet = BigResnet.resnet50(num_classes=numclass)
elif args.model == "WRN1602":
    studentnet = WRN.wideresnet1602(num_classes=numclass)
elif args.model == "WRN4002":
    studentnet = WRN.wideresnet4002(num_classes=numclass)
elif args.model == "densenet40":
    studentnet = Densenet.densenet_40(num_classes=numclass)
elif args.model == "densenet100_12":
    studentnet = Densenet.densenet_100_12(num_classes=numclass)
elif args.model == "densenet100_24":
    studentnet = Densenet.densenet_100_24(num_classes=numclass)
# 添加对 ResNet8x4 和 ResNet32x4 的支持
elif args.model == "resnet8x4":
    studentnet = Resnet.resnet8x4(num_classes=numclass)
elif args.model == "resnet32x4":
    studentnet = Resnet.resnet32x4(num_classes=numclass)
elif args.model == "resnet110":
    studentnet = BigResnet.resnet110(num_classes=numclass)
elif args.model == "vgg8":
    studentnet = VGG.vgg8(num_classes=numclass)
elif args.model == "vgg13":
    studentnet = VGG.vgg13(num_classes=numclass)
elif args.model == "mobilenetv2":  # 添加对 MobileNetV2 的支持
    from models.mobilenetv2 import MobileNetV2CIFAR
    studentnet = MobileNetV2CIFAR(num_classes=numclass)

studentnet = nn.DataParallel(studentnet).cuda()
# PATH = "./checkpoints/cifar100/alone" + "WRN4002"+ ".pth"
# studentnet.load_state_dict(torch.load(PATH), False)
# print("Waiting Test!")
# with torch.no_grad():
#     total = 0.0
#     correct = 0
#     for data in testloader:
#         studentnet.eval()
#         images, labels = data
#         images, labels = images.cuda(), labels.cuda()
#         studentout, _ = studentnet(images)
#         total += float(labels.size(0))
#         _, predicted = torch.max(studentout.data, 1)
#         correct += float(predicted.eq(labels.data).cpu().sum())
#
#     print('s Test Acc: %.2f%%' % (100 * correct / total))

# ce是交叉熵损失函数
def CEloss(modelout, target):
    N = target.size(0)
    C = numclass
    labels = torch.full(size=(N, C), fill_value=0, dtype=torch.float).cuda()  
    labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)  
    log_prob = torch.nn.functional.log_softmax(modelout, dim=1)
    loss = -torch.sum(log_prob * labels) / N
    return loss




def aloneTrain_main():
    filetime = str(time.strftime("%Y__%m-%d__%H_%M_%S", time.localtime()))
    #produce xxx.pth,user need to change the studentnet to achieve your aim.
    # 确保保存路径存在
    os.makedirs("./checkpoints/cifar100", exist_ok=True)
    os.makedirs("./result/cifar100/baseline", exist_ok=True)  # 确保结果路径也存在
    # 打开保存训练日志的文件
    baseKD_file = open("./result/cifar100/baseline/Alonestudent"+ str(args.model)+str(filetime) + '.txt', 'w')

    # 优化器sgd
    student_optimizer = optim.SGD(studentnet.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
    for epoch in range(args.epoch):

        if epoch in [150,225]:
            for param_group in student_optimizer.param_groups:
                param_group['lr'] /= 10

        studentnet.train()
        sum_loss, total = 0.0, 0.0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            studentout,_ = studentnet(inputs)
            loss = torch.FloatTensor([0.]).cuda()
            loss = loss + CEloss(studentout, labels)
            sum_loss += loss.item()
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            total += float(labels.size(0))
            _, predicted = torch.max(studentout.data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%%' %
                  (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100 * correct / total))
            baseKD_file.write('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%%' %
                              (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100 * correct / total))
            baseKD_file.write('\n')

        print("Waiting Test!")
        with torch.no_grad():
            total = 0.0
            correct = 0
            for data in testloader:
                studentnet.eval()
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                studentout,_ = studentnet(images)
                total += float(labels.size(0))
                _, predicted = torch.max(studentout.data, 1)
                correct+= float(predicted.eq(labels.data).cpu().sum())


            print('s Test Acc: %.2f%%' % (100 * correct / total))

            print('Filename:', str(args.model) + filetime + '.txt')


            baseKD_file.write('s Test Acc: %.2f%%' % (100 * correct / total))
            baseKD_file.write('\n')

        torch.save(studentnet.state_dict(), "./checkpoints/cifar100/" + 'alone' + str(args.model) + ".pth")



    baseKD_file.close()





aloneTrain_main()