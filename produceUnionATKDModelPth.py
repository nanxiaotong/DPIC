import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from models import Resnet as Resnet
from models import wideResnet as WRN
from models import densenet as Densenet
from models import BigResnet
import torch.nn.functional as F
import numpy as np
import time
from models.autoaugment import CIFAR10Policy
from models.cutout import Cutout
import torch.utils.data as data
from models import VGG

filetime = str(time.strftime("%Y__%m-%d__%H_%M_%S", time.localtime()))
np.random.seed(0)
torch.manual_seed(0)
parser = argparse.ArgumentParser(description='produce KDmodel KDxxx_from_xxx.pth')
parser.add_argument('--model', default="resnet20", type=str, help="resnet20|WRN1602")
parser.add_argument('--teachermodel', default="resnet32", type=str, help="resnet20|resnet32|resnet56|WRN1602|WRN4002")
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar100")
parser.add_argument('--epoch', default=300, type=int, help="training epochs")
parser.add_argument('--loss_coefficient', default=0.3, type=float)
parser.add_argument('--feature_loss_coefficient', default=0.03, type=float)
parser.add_argument('--dataset_path', default="data", type=str)
parser.add_argument('--autoaugment', default=True, type=bool)
parser.add_argument('--temperature', default=4.8, type=float)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--init_lr', default=0.01, type=float)
parser.add_argument('--alpha', default=0.7, type=float)
parser.add_argument('--epochp', default=30, type=int)
parser.add_argument('--balance_func', default=1, type=int)
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# '0,1,2,3,4,5'
torch.set_printoptions(threshold=np.inf)

global numclass
global trainloader
global testloader

if args.dataset == "cifar100":
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                      transforms.RandomHorizontalFlip(), CIFAR10Policy(), transforms.ToTensor(),
                                      Cutout(n_holes=1, length=16),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

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




global baseteachernet
global studentnet
global pretrained_studentnet

if args.model == "resnet20":
    studentnet = Resnet.resnet20(num_classes=numclass)
    pretrained_studentnet = Resnet.resnet20(num_classes=numclass)
elif args.model == "resnet32":
    studentnet = Resnet.resnet32(num_classes=numclass)
    pretrained_studentnet = Resnet.resnet32(num_classes=numclass)
elif args.model == "resnet56":
    studentnet = Resnet.resnet56(num_classes=numclass)
    pretrained_studentnet = Resnet.resnet56(num_classes=numclass)
elif args.model == "resnet18":
    studentnet = BigResnet.resnet18(num_classes=numclass)
    pretrained_studentnet = BigResnet.resnet18(num_classes=numclass)
elif args.model == "resnet34":
    studentnet = BigResnet.resnet34(num_classes=numclass)
    pretrained_studentnet = BigResnet.resnet34(num_classes=numclass)
elif args.model == "WRN1602":
    studentnet = WRN.wideresnet1602(num_classes=numclass)
    pretrained_studentnet = WRN.wideresnet1602(num_classes=numclass)
elif args.model == "WRN4001":
    studentnet = WRN.wideresnet4001(num_classes=numclass)
    pretrained_studentnet = WRN.wideresnet4001(num_classes=numclass)
elif args.model == "WRN4002":
    studentnet = WRN.wideresnet4002(num_classes=numclass)
    pretrained_studentnet = WRN.wideresnet4002(num_classes=numclass)
elif args.model == "densenet40":
    studentnet = Densenet.densenet_40(num_classes=numclass)
    pretrained_studentnet = Densenet.densenet_40(num_classes=numclass)
elif args.model == "densenet100_12":
    studentnet = Densenet.densenet_100_12(num_classes=numclass)
    pretrained_studentnet = Densenet.densenet_100_12(num_classes=numclass)
elif args.model == "densenet100_24":
    studentnet = Densenet.densenet_100_24(num_classes=numclass)
    pretrained_studentnet = Densenet.densenet_100_24(num_classes=numclass)
elif args.model == "resnet110":
    studentnet = BigResnet.resnet110(num_classes=numclass)
    pretrained_studentnet = BigResnet.resnet110(num_classes=numclass)
elif args.model == "vgg8":
    studentnet = VGG.vgg8(num_classes=numclass)
    pretrained_studentnet = VGG.vgg8(num_classes=numclass)
elif args.model == "vgg13":
    studentnet = VGG.vgg13(num_classes=numclass)
    pretrained_studentnet = VGG.vgg13(num_classes=numclass)
elif args.model == "resnet32x4":
    studentnet = Resnet.resnet32x4(num_classes=numclass)
    pretrained_studentnet = Resnet.resnet32x4(num_classes=numclass)
elif args.model == "resnet8x4":
    studentnet = Resnet.resnet8x4(num_classes=numclass)
    pretrained_studentnet = Resnet.resnet8x4(num_classes=numclass)
elif args.model == "mobilenetv2":  # 添加对 MobileNetV2 的支持
    from models.mobilenetv2 import MobileNetV2CIFAR
    studentnet = MobileNetV2CIFAR(num_classes=numclass)
    pretrained_studentnet = MobileNetV2CIFAR(num_classes=numclass)

if args.teachermodel == "resnet20":
    baseteachernet = Resnet.resnet20(num_classes=numclass)
elif args.teachermodel == "resnet32":
    baseteachernet = Resnet.resnet32(num_classes=numclass)
elif args.teachermodel == "resnet56":
    baseteachernet = Resnet.resnet56(num_classes=numclass)
elif args.teachermodel == "resnet18":
    baseteachernet = BigResnet.resnet18(num_classes=numclass)
elif args.teachermodel == "resnet34":
    baseteachernet = BigResnet.resnet34(num_classes=numclass)
elif args.teachermodel == "WRN1602":
    baseteachernet = WRN.wideresnet1602(num_classes=numclass)
elif args.teachermodel == "WRN4001":
    baseteachernet = WRN.wideresnet4001(num_classes=numclass)
elif args.teachermodel == "WRN4002":
    baseteachernet = WRN.wideresnet4002(num_classes=numclass)
elif args.teachermodel == "densenet40":
    baseteachernet = Densenet.densenet_40(num_classes=numclass)
elif args.teachermodel == "densenet100_12":
    baseteachernet = Densenet.densenet_100_12(num_classes=numclass)
elif args.teachermodel == "densenet100_24":
    baseteachernet = Densenet.densenet_100_24(num_classes=numclass)
elif args.teachermodel == "resnet110":
    baseteachernet = BigResnet.resnet110(num_classes=numclass)
elif args.teachermodel == "vgg8":
    baseteachernet = VGG.vgg8(num_classes=numclass)
elif args.teachermodel == "vgg13":
    baseteachernet = VGG.vgg13(num_classes=numclass)
elif args.teachermodel == "resnet32x4":
    baseteachernet = Resnet.resnet32x4(num_classes=numclass)
elif args.teachermodel == "resnet8x4":
    baseteachernet = Resnet.resnet8x4(num_classes=numclass)
elif args.teachermodel == "mobilenetv2":  # 添加对 MobileNetV2 的支持
    from models.mobilenetv2 import MobileNetV2CIFAR
    baseteachernet = MobileNetV2CIFAR(num_classes=numclass)

epochp = args.epochp


studentnet = nn.DataParallel(studentnet).cuda()
baseteachernet = nn.DataParallel(baseteachernet).cuda()
pretrained_studentnet= nn.DataParallel(pretrained_studentnet).cuda()


def CEloss(modelout, target):
    N = target.size(0)
    C = numclass
    labels = torch.full(size=(N, C), fill_value=0, dtype=torch.float).cuda() 
    labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)  
    log_prob = torch.nn.functional.log_softmax(modelout, dim=1)
    loss = -torch.sum(log_prob * labels) / N
    return loss

def SoftCEloss(modelout, target):
    N = target.size(0)
    log_prob = torch.nn.functional.log_softmax(modelout, dim=1)
    target = F.softmax(target, dim=1)
    loss = -torch.sum(log_prob * target) / N
    return loss

def distillation_old(y, labels, teacher_scores,temp, alpha):
    return SoftCEloss(y / temp,teacher_scores / temp)*(temp * temp * 2.0 * alpha) + \
           CEloss(y, labels) * (1. - alpha)


def at_loss(x, y):
    return (x - y).pow(2).mean()

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))



def Union_distillation_attention(student_output, stuAttList, pretrained_stuAttList, labels, teacher_output, teaAttList, temp, alpha):

    teacher_attention_loss=sum([at_loss(at(x) / temp, at(y).detach() / temp) for x, y in zip(stuAttList, teaAttList)])
    pretrained_attention_loss=sum([at_loss(at(x) / temp, at(y).detach() / temp) for x, y in zip(stuAttList, pretrained_stuAttList)])
    return torch.pow(torch.mul(teacher_attention_loss,pretrained_attention_loss),0.5)+\
           SoftCEloss(student_output / temp,teacher_output.detach() / temp)*(temp * temp * alpha) + \
           CEloss(student_output,labels)

def Union_distillation_attention2(student_output, stuAttList, pretrained_stuAttList, labels, teacher_output, teaAttList, temp, alpha):

    teacher_attention_loss=sum([at_loss(at(x) / temp, at(y).detach() / temp) for x, y in zip(stuAttList, teaAttList)])
    pretrained_attention_loss=sum([at_loss(at(x) / temp, at(y).detach() / temp) for x, y in zip(stuAttList, pretrained_stuAttList)])
    return torch.mul(torch.pow(teacher_attention_loss,0.5),torch.log(pretrained_attention_loss))+\
           SoftCEloss(student_output / temp,teacher_output.detach() / temp)*(temp * temp * alpha) + \
           CEloss(student_output,labels)


def distillation_attention(student_output,student_mid, labels,teacher_output,teacher_mid, temp, alpha=0.5):
    return sum([at_loss(at(x)/temp, at(y).detach()/temp) for x, y in zip(student_mid,teacher_mid)])*alpha/2\
           +SoftCEloss(student_output / temp,teacher_output / temp)*(temp * temp * alpha) + \
            CEloss(student_output,labels)


def pre_distill():
    pretrained_student_optimizer = optim.SGD(pretrained_studentnet.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
    for epoch in range(epochp):
        # baseKD_file = open("./result/cifar100/UnionATKD/" + str(args.model) + "PreATfrom" + str(args.teachermodel) + filetime + '.txt', 'w')
        # baseKD_file.write('[epochp:%d] T: %.03f | alpha: %.2f%%' %(epochp,args.temperature,args.epochp))
        pretrained_studentnet.train()
        pre_sum_loss, pre_total = 0.0, 0.0
        pre_correct = 0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            teacherout,baseteachernet_attention = baseteachernet(inputs)
            teacherout = teacherout.detach()


            #Proximal Teacher Training
            pretrained_studentout,pretrained_student_attention = pretrained_studentnet(inputs)
            pre_loss = torch.FloatTensor([0.]).cuda()
            pre_loss = pre_loss + distillation_attention(pretrained_studentout, pretrained_student_attention, labels,
                                                         teacherout, baseteachernet_attention, temp=args.temperature,
                                                         alpha=args.alpha)
            pretrained_student_optimizer.zero_grad()
            pre_loss.backward(retain_graph=True)
            pretrained_student_optimizer.step()
            pre_sum_loss += pre_loss.item()
            pre_total += float(labels.size(0))
            _, pretrained_predicted = torch.max(pretrained_studentout.data, 1)
            pre_correct += float(pretrained_predicted.eq(labels.data).cpu().sum())

            print('[epoch:%d, iter:%d]  preLoss: %.03f | pretrained_stuAcc: %.2f%%' %
                  (epoch + 1, (i + 1 + epoch * length),pre_sum_loss / (i + 1), 100 * pre_correct / pre_total)
                  )

        print("Waiting Test!")
        with torch.no_grad():
            total = 0.0
            correct = 0

            for data in testloader:
                pretrained_studentnet.eval()
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                studentout, _ = pretrained_studentnet(images)
                total += float(labels.size(0))
                _, predicted = torch.max(studentout.data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())

            print('prestu Test Acc: %.2f%% ' % (100 * correct / total))



        print('pretrain over!')




def distill_main():


    pre_distill()
    if args.dataset == 'cifar100':
        # 确保路径存在
        output_dir = "./result/cifar100/UnionATKD/"
        os.makedirs(output_dir, exist_ok=True)  # 自动创建目录

        baseKD_file = open(output_dir + str(args.model) + "UnionATfrom" + str(args.teachermodel) + filetime + '.txt',
                           'w')
        PATH = "./checkpoints/cifar100/alone" + args.teachermodel + ".pth"

    baseteachernet.load_state_dict(torch.load(PATH), False)

    student_optimizer = optim.SGD(studentnet.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
    pretrained_student_optimizer = optim.SGD(pretrained_studentnet.parameters(), lr=args.init_lr, weight_decay=5e-4, momentum=0.9)
    for epoch in range(args.epoch-epochp):
        if epoch in [150,225]:
            for param_group in student_optimizer.param_groups:
                param_group['lr'] /= 10
        if epoch in [150-epochp, 225-epochp]:
            for param_group in pretrained_student_optimizer.param_groups:
                param_group['lr'] /= 10
        studentnet.train()
        pretrained_studentnet.train()

        sum_loss, total = 0.0, 0.0
        correct = 0
        pre_sum_loss, pre_total = 0.0, 0.0
        pre_correct = 0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            teacherout,baseteachernet_attention = baseteachernet(inputs)
            teacherout = teacherout.detach()


            #Proximal Teacher Training
            pretrained_studentout,pretrained_student_attention = pretrained_studentnet(inputs)


            pre_loss = torch.FloatTensor([0.]).cuda()
            pre_loss = pre_loss + distillation_attention(pretrained_studentout, pretrained_student_attention, labels,
                                                         teacherout, baseteachernet_attention, temp=args.temperature,
                                                         alpha=args.alpha)

            pretrained_student_optimizer.zero_grad()#我算一个loss，把optimizer置0，然后反向传播，不就是对应的网络了
            pre_loss.backward(retain_graph=True)
            pretrained_student_optimizer.step()
            pre_sum_loss += pre_loss.item()
            pre_total += float(labels.size(0))
            _, pretrained_predicted = torch.max(pretrained_studentout.data, 1)
            pre_correct += float(pretrained_predicted.eq(labels.data).cpu().sum())

            #student training
            studentout,studentnet_attention = studentnet(inputs)
            loss = torch.FloatTensor([0.]).cuda()


            if epoch<=epochp:
            # at the begining , the Proximal Teacher may be wrong, we add it after a period of time (here we select P epochs too, and you can choose another time period).
                loss = loss + distillation_attention(studentout,studentnet_attention,
                                                             labels,
                                                             teacherout, baseteachernet_attention,
                                                             temp=args.temperature,
                                                             alpha=args.alpha)
            else:
                if args.balance_func == 1:
                    loss = loss + Union_distillation_attention(studentout,studentnet_attention,
                                                           pretrained_student_attention, labels,
                                                           teacherout,baseteachernet_attention,
                                                           temp=args.temperature, alpha=args.alpha)
                elif args.balance_func == 2:
                    loss = loss + Union_distillation_attention2(studentout,studentnet_attention,
                                                               pretrained_student_attention, labels,
                                                               teacherout,baseteachernet_attention,
                                                               temp=args.temperature, alpha=args.alpha)


            sum_loss += loss.item()
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            total += float(labels.size(0))
            _, predicted = torch.max(studentout.data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            print('[epoch:%d, iter:%d] Loss: %.03f | stuAcc: %.2f%% | preLoss: %.03f | pretrained_stuAcc: %.2f%%' %
                  (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100 * correct / total,pre_sum_loss / (i + 1), 100 * pre_correct / pre_total)
                  )




            baseKD_file.write('[epoch:%d, iter:%d] Loss: %.03f | stuAcc: %.2f%% | preLoss: %.03f | pretrained_stuAcc: %.2f%%' %
                  (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100 * correct / total,pre_sum_loss / (i + 1), 100 * pre_correct / pre_total)
                  )
            baseKD_file.write('\n')
        print("Waiting Test!")
        with torch.no_grad():
            total = 0.0
            correct = 0
            pre_total = 0.0
            pre_correct = 0
            for data in testloader:
                studentnet.eval()
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                studentout,_ = studentnet(images)
                total += float(labels.size(0))
                _, predicted = torch.max(studentout.data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())

                pretrained_studentnet.eval()

                pre_studentout,_ = pretrained_studentnet(images)
                pre_total += float(labels.size(0))
                _, pretrained_predicted = torch.max(pre_studentout.data, 1)
                pre_correct += float(pretrained_predicted.eq(labels.data).cpu().sum())
            print('stu Test Acc: %.2f%% | prestu Test Acc: %.2f%%' % (100 * correct / total,100 * pre_correct / pre_total))

            baseKD_file.write('stu Test Acc: %.2f%% | prestu Test Acc: %.2f%%' % (100 * correct / total,100 * pre_correct / pre_total))
            baseKD_file.write('\n')


        torch.save(pretrained_studentnet.state_dict(),"./checkpoints/cifar100/" + 'ATKD_' + str(args.model) + '_from_' + str(args.teachermodel) + ".pth")
        torch.save(studentnet.state_dict(),"./checkpoints/cifar100/" + 'UnionATKD'+'_' + str(args.model) + '_from_' + str(args.teachermodel) + ".pth")
        print('Filename:', str(args.model) +"from"+str(args.teachermodel)+ filetime+ '.txt')
        baseKD_file.write('epochp:%d' % (epochp))
        baseKD_file.write('\n')
    baseKD_file.close()


distill_main()

