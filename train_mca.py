import torch
import torch.nn as nn

import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from libtiff import TIFF
# from PIL import Image
import numpy as np
from scipy.io import loadmat
import cv2
from scipy import io
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from collections import Counter
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import math
import os
# import h5p y
import time
import datetime as dt
from mcaEasyCalc import Network
from PIL import Image
# from PANBranch import NFNet
# from EffNetV2_ceshi import EfficientNetV2, ShuffleNetV2, shufflenet_v2
# import pyfftw
# from EffNetV2_ceshi import efficientnetv2
start_time = dt.datetime.now().strftime('%F %T')
print("程序开始运行时间：" + start_time)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定使用哪块GPU运行以下程序‘0’代表第一块，‘1’代表第二块


import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())





# 设计随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     # random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)


# 网络超参数
EPOCH = 30
BATCH_SIZE = 64
# BATCH_SIZE1 = 1000
# LR = 0.0002
LR = 0.0005# Batch*4,LR*2
Train_Rate = 0.05 # 训练集和测试集按比例拆分,使用列表堆叠
ms4_np = loadmat("./dataset/Muufl/data_HS_LR.mat")
# ms4_np = h5py.File('./MUUFL/data_HS_LR.mat')
ms4_np = np.transpose(ms4_np['hsi_data']).transpose((1, 2, 0))
# ms4_np = np.array(ms4_np, dtype='uint32')  # image3的标签类型为float，此处要强制转换为uint8类型
print('ms4_np:', np.shape(ms4_np))

pan_np = loadmat("./dataset/Muufl/data_SAR_HR.mat")
# pan_tif = h5py.File('./MUUFL/data_SAR_HR.mat')
pan_np = np.transpose(pan_np['lidar_data']).transpose((1, 2, 0))
# pan_np = np.array(pan_np, dtype='uint8')  # image3的标签类型为float，此处要强制转换为uint8类型
print('pan_np:', np.shape(pan_np))

label_np_float = loadmat("./dataset/Muufl/gt.mat")
# label_np_float = h5py.File('./MUUFL/gt.mat')
label_np = np.transpose(label_np_float['labels'])
# label_np = np.array(label_np, dtype='uint8')  # image3的标签类型为float，此处要强制转换为uint8类型
print('label shape:', np.shape(label_np))

# ms4与pan图补零
Ms4_patch_size = 16  # ms4截块的边长
Interpolation = cv2.BORDER_REFLECT_101
# cv2.BORDER_REPLICATE： 进行复制的补零操作;
# cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
# cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
# cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdegh|abcdefgh|abcdefg;

top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的ms4图的形状：', np.shape(ms4_np))

# Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
Pan_patch_size = Ms4_patch_size  # pan截块的边长
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 1), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 1), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
print('补零后的pan图的形状：', np.shape(pan_np))

# 按类别比例拆分数据集
# label_np=label_np.astype(np.uint8)
label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255
# label_np = label_np      # shanghai数据集用这个得，其余数据集-1
# np.unique() 函数 去除其中重复的元素 ，arr：输入数组,return_counts：如果为 true，返回去重数组中的元素在原数组中的出现次数
label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
# print('类标：', label_element)
# print('各类样本数：', element_count)
Categories_Number = len(label_element) - 1  # 数据的类别数
# print('标注的类别数：', Categories_Number)
label_row, label_column = np.shape(label_np)  # 获取标签图的行、列

'''归一化图片'''

def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image


# .tolist是将数组转为list的格式
ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)

count = 0
for row in range(label_row):  # 行
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != -2:
            ground_xy[int(label_np[row][column])].append([row, column])
        # if label_np[row][column] != 0:          # shanghai数据集用这个得，其余数据集255
        #     ground_xy[int(label_np[row][column]-1)].append([row, column])      # shanghai数据集-1，其余数据集不减

        # 标签内打乱
for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    # np.random,shuffle作用就是重新排序返回一个随机序列作用类似洗牌
    np.random.shuffle(shuffle_array)
    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    # print('aaa', categories_number)
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

# 训练数据与测试数据，数据集内打乱
shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

# print('训练样本数：', len(label_train))
# print('测试样本数：', len(label_test))

# 数据归一化
ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)
# mshpan = to_tensor(mshpan_np)
# pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
# mshpan = np.expand_dims(mshpan, axis=0)  # 二维数据进网络前要加一维
pan = np.array(pan).transpose((2, 0, 1))  # 调整通道
ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

# 转换类型
ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)
# mshpan = torch.from_numpy(mshpan).type(torch.FloatTensor)


class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, xy, cut_size):
    # def __init__(self, MS4, Pan, MSHPAN, Label, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        # self.train_data3 = MSHPAN
        self.train_labels = Label
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        # image_mshpan = self.train_data3[:, x_pan:x_pan + self.cut_pan_size,
        #                y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, target, locate_xy
        # return image_ms, image_pan, image_mshpan, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    # def __init__(self, MS4, Pan, MSHPAN, xy, cut_size):
    def __init__(self, MS4, Pan, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        # self.train_data3 = MSHPAN
        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        # image_mshpan = self.train_data3[:, x_pan:x_pan + self.cut_pan_size,
        #                y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        # return image_ms, image_pan, image_mshpan, locate_xy
        return image_ms, image_pan, locate_xy

    def __len__(self):
        return len(self.gt_xy)


# cnn = ResNet18()
cnn = Network(64,2,11)
# cnn = EfficientNetV2()
# cnn = shufflenet_v2()
# print(cnn)  # net architecture

# ================#
#     GPU并行    #
# ===============#
# if torch.cuda.device_count() > 1:
#     print("===== Let's use", torch.cuda.device_count(), "GPUs! =====")
#     cnn = nn.DataParallel(cnn, device_ids=[0, 1]).cuda()

# 调整学习率
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # LR = LR * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        # param_group['lr'] = param_group['lr'] * 0.99
        param_group['lr'] = param_group['lr'] * 0.9995 # 可以调整


# 参数初始化方法 # 可以改变
# for m in EfficientNetV2.modules(cnn):
# for m in ShuffleNetV2.modules(cnn):
for m in Network.modules(cnn):
# for m in ResNet18.modules(cnn):
    if isinstance(m, (nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)

cnn.cuda()
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.00001) # 可以修改，+decay
scheduler = lr_scheduler.StepLR(optimizer, step_size=21 , gamma=0.05) # 可以修改
# optimizer  = torch.optim.RMSprop(cnn.parameters(), lr=LR, alpha=0.9)

# loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted


loss_func = nn.CrossEntropyLoss()


train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = Data.DataLoader(dataset=all_data, batch_size=BATCH_SIZE * 20, shuffle=False, num_workers=0)
print("ms4 = ", ms4.shape)
print("pan = ", pan.shape)
print("label_test = ", label_test.shape)
print("label_train = ", label_train.shape)

trainstart = time.time()
for epoch in range(EPOCH):
    valid_batch = iter(test_loader)  # 验证集迭代器
    # for step, (ms, pan, MSHpan, label, _) in enumerate(
    for step, (ms, pan, label, _) in enumerate(
            train_loader):  # gives batch data, normalize x when iterate train_loader
        # print(label)
        cnn.train()
        ms = ms.cuda()
        pan = pan.cuda()
        # MSHpan = MSHpan.cuda()
        label = label.cuda()

        output = cnn(ms, pan)  # cnn output
        # output = cnn(ms, pan, MSHpan)  # cnn output
        # print(output)
        loss = loss_func(output, label)  # cross entropy loss
        # print('AAA', loss)
        # pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
        # accuracy = (pred_y == label).sum().item() / float(label.size(0))
        # print(accuracy)
        # adjust_learning_rate(optimizer, epoch)

        # refresh the optimizer，清空过往梯度
        optimizer.zero_grad()  # clear gradients for this training step
        # compute gradients and take step，反向传播，计算当前梯度
        loss.backward()  # backpropagation, compute gradients
        # 根据梯度更新网络参数
        optimizer.step()  # apply gradients
        # print('| train loss: %.4f' % loss.item())
        cnn.eval()
        if step % 100 == 0:
            # print('| train loss: %.4f' % loss.item())
            # ms4_test1, pan_test1, MSHpan_test1, label_test1, _ = next(valid_batch)
            ms4_test1, pan_test1, label_test1, _ = next(valid_batch)
            ms4_test1 = ms4_test1.cuda()
            pan_test1 = pan_test1.cuda()
            # MSHpan_test1 = MSHpan_test1.cuda()
            label_test1 = label_test1.cuda()

            with torch.no_grad():
                # test_output = cnn(ms4_test1, pan_test1, MSHpan_test1)
                test_output = cnn(ms4_test1, pan_test1)
            # print('zz',test_output)
            pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()

            # tensor 对uint8求和要使用torch中的sum()方法，而不是调用python中的函数sum()！！！
            accuracy = (pred_y == label_test1).sum().item() / float(label_test1.size(0))
            # print('bbbbb', label_test)
            print('Epoch:%3d' % epoch, '|| step: %3d' % step, '|| train loss: %.4f' % loss.item(),
                  '|| test accuracy: %.4f' % accuracy)
        adjust_learning_rate(optimizer, epoch)
    # print('Epoch: %d ， Lr: %.8f' % (epoch, scheduler.get_lr()[0]))
    scheduler.step()
# torch.save(cnn.state_dict(), 'net_param.pkl')
trainend = time.time()
# 保存模型
torch.save(cnn, "./out_data/model/MUUFL_model.pkl")
end_time = dt.datetime.now().strftime('%F %T')
print("程序结束运行时间：" + end_time)


# 加载模型
begin_test = time.time()
cnn2 = torch.load("./out_data/model/MUUFL_model.pkl")

cnn2.cuda()
out_color = np.zeros((220, 325, 3))
out_color_bufen=np.zeros((220,325,3))
l = 0
y_pred = []
cnn2.eval()

for step, (ms, pan, label, gt_xy) in enumerate(test_loader):
    l = l + 1
    ms = ms.cuda()
    pan = pan.cuda()
    # MSHpan = MSHpan.cuda()
    label = label.cuda()
    with torch.no_grad():
        # output = cnn2(ms, pan, MSHpan)  # cnn output
        output = cnn2(ms, pan)  # cnn output
    pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
    if l == 1:
        y_pred = pred_y.cpu().numpy()
    else:
        y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis=0)
end_test = time.time()
test_time = end_test - begin_test
print('test_loader长度:', len(test_loader))
con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred)
print('con_mat', con_mat)

# 计算性能参数
all_acr = 0
p = 0
column = np.sum(con_mat, axis=0)  # 列求和
line = np.sum(con_mat, axis=1)  # 行求和
for i, clas in enumerate(con_mat):
    precise = clas[i]
    all_acr = precise + all_acr
    acr = precise / column[i]
    recall = precise / line[i]
    f1 = 2 * acr * recall / (acr + recall)
    temp = column[i] * line[i]
    p = p + temp
    # print('PRECISION:',acr,'||RECALL:',recall,'||F1:',f1)#查准率 #查全率 #F1
    print("第 %d 类: || 准确率: %.7f || 召回率: %.7f || F1: %.7f " % (i, acr, recall, f1))
OA = np.trace(con_mat) / np.sum(con_mat)
print('OA:', OA)

AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))  # axis=1 每行求和
print('AA:', AA)

Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
Kappa = (OA - Pc) / (1 - Pc)
print('Kappa:', Kappa)

# 生成着色图
class_count = np.zeros(12)
# for step, (ms, pan, gt_xy) in enumerate(all_data_loader):
for step, (ms, pan, gt_xy) in enumerate(all_data_loader):
    ms = ms.cuda()
    pan = pan.cuda()

    with torch.no_grad():
        output = cnn2(ms, pan)  # cnn output

    pred_y = torch.argmax(output, 1)
    pred_y_numpy = pred_y.cpu().numpy()
    gt_xy = gt_xy.numpy()
    for k in range(len(gt_xy)):
        if pred_y_numpy[k] == 0:
            class_count[0] = class_count[0] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 127, 0]
        elif pred_y_numpy[k] == 1:
            class_count[1] = class_count[1] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
        elif pred_y_numpy[k] == 2:
            class_count[2] = class_count[2] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
        elif pred_y_numpy[k] == 3:
            class_count[3] = class_count[3] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 204, 255]
        elif pred_y_numpy[k] == 4:
            class_count[4] = class_count[4] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [51, 0, 255]
        elif pred_y_numpy[k] == 5:
            class_count[5] = class_count[5] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [204, 0, 0]
        elif pred_y_numpy[k] == 6:
            class_count[6] = class_count[6] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [204, 0, 102]
        elif pred_y_numpy[k] == 7:
            class_count[7] = class_count[7] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [153, 127, 255]
        elif pred_y_numpy[k] == 8:
            class_count[8] = class_count[8] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 102, 204]
        elif pred_y_numpy[k] == 9:
            class_count[9] = class_count[9] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
        elif pred_y_numpy[k] == 10:
            class_count[10] = class_count[10] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [102, 25, 204]
        elif pred_y_numpy[k] == -2:
            class_count[11] = class_count[11] + 1
            out_color[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 0]

save_out_color_path = './out_data/result/out' + 'MUUFL_quanse.png'
cv2.imwrite(save_out_color_path, out_color)
for step, (ms, pan, label, gt_xy) in enumerate(test_loader):
    ms = ms.cuda()
    pan = pan.cuda()
    # MSHpan = MSHpan.cuda()
    with torch.no_grad():
        # output = cnn2(ms, pan, MSHpan)  # cnn output
        output = cnn2(ms, pan)  # cnn output

    pred_y = torch.max(output, 1)[1].cuda().data.squeeze()
    pred_y_numpy = pred_y.cpu().numpy()
    gt_xy = gt_xy.numpy()
    for k in range(len(gt_xy)):
        if pred_y_numpy[k] == 0:
            class_count[0] = class_count[0] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 127, 0]
        elif pred_y_numpy[k] == 1:
            class_count[1] = class_count[1] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
        elif pred_y_numpy[k] == 2:
            class_count[2] = class_count[2] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
        elif pred_y_numpy[k] == 3:
            class_count[3] = class_count[3] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 204, 255]
        elif pred_y_numpy[k] == 4:
            class_count[4] = class_count[4] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [51, 0, 255]
        elif pred_y_numpy[k] == 5:
            class_count[5] = class_count[5] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [204, 0, 0]
        elif pred_y_numpy[k] == 6:
            class_count[6] = class_count[6] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [204, 0, 102]
        elif pred_y_numpy[k] == 7:
            class_count[7] = class_count[7] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [153, 127, 255]
        elif pred_y_numpy[k] == 8:
            class_count[8] = class_count[8] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 102, 204]
        elif pred_y_numpy[k] == 9:
            class_count[9] = class_count[9] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
        elif pred_y_numpy[k] == 10:
            class_count[10] = class_count[10] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [102, 25, 204]
        elif pred_y_numpy[k] == -2:
            class_count[11] = class_count[11] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 0]
for step, (ms, pan, label, gt_xy) in enumerate(train_loader):
    ms = ms.cuda()
    pan = pan.cuda()
    label = label.cuda()
    with torch.no_grad():
        output = cnn2(ms, pan)  # cnn output

    pred_y = torch.argmax(output, 1)
    pred_y_numpy = pred_y.cpu().numpy()
    gt_xy = gt_xy.numpy()
    for k in range(len(gt_xy)):
        if pred_y_numpy[k] == 0:
            class_count[0] = class_count[0] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 127, 0]
        elif pred_y_numpy[k] == 1:
            class_count[1] = class_count[1] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 0]
        elif pred_y_numpy[k] == 2:
            class_count[2] = class_count[2] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [255, 255, 0]
        elif pred_y_numpy[k] == 3:
            class_count[3] = class_count[3] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 204, 255]
        elif pred_y_numpy[k] == 4:
            class_count[4] = class_count[4] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [51, 0, 255]
        elif pred_y_numpy[k] == 5:
            class_count[5] = class_count[5] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [204, 0, 0]
        elif pred_y_numpy[k] == 6:
            class_count[6] = class_count[6] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [204, 0, 102]
        elif pred_y_numpy[k] == 7:
            class_count[7] = class_count[7] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [153, 127, 255]
        elif pred_y_numpy[k] == 8:
            class_count[8] = class_count[8] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 102, 204]
        elif pred_y_numpy[k] == 9:
            class_count[9] = class_count[9] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 255, 255]
        elif pred_y_numpy[k] == 10:
            class_count[10] = class_count[10] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [102, 25, 204]
        elif pred_y_numpy[k] == -2:
            class_count[11] = class_count[11] + 1
            out_color_bufen[gt_xy[k][0]][gt_xy[k][1]] = [0, 0, 0]

print(class_count)
save_out_color_bufen_path = './out_data/result/out' + 'MUUFL_bufen.png'
cv2.imwrite(save_out_color_bufen_path, out_color_bufen)
print(save_out_color_bufen_path)
end_time = dt.datetime.now().strftime('%F %T')
print("end time:" + end_time)
print("test time: %.2f S" % test_time)