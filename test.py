import torch
import time
import math
import timm
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch.nn as nn

# a = [1, 2, 3, 4, 5]
# b = a[:-1]
# c = a[:-2]
# print(b)
# print(c)
#
# for blk in a[:-2]:
#     print(blk)

# target = torch.randint(5, (3,), dtype=torch.int64)
# print(target)
labels_a = torch.randint(10, (16, 6, 576, 576), dtype=torch.int64)
labels_b = torch.randint(5, (16,6, 576, 64), dtype=torch.int64)
# # labels_a = torch.randint(99, (16, 6, 576, 576), dtype=torch.int64)
# # labels_b = torch.randint(99, (16, 6, 576, 576), dtype=torch.int64)
c = labels_a @ labels_b
print(c.size())
#
# c = c[:, :, 0, :]
# # c = c[0, :]
#
# _, part_inx = c.max(2)
#
# print(c)
#
# print(part_inx)
# #
# new = []
# for i in range(part_inx.size(0)):
#     # print(part_inx[i])
#     a = part_inx[i]
#     # print(a)
#     for index in range(a.size(0)):
#         b = a[index]
#         new.append(torch.cat([c[a, 0:b], c[a, b + 1:]]))
#         # print(index)
#
# # a = c[part_inx]
#
# print(new)

# x = torch.randint(99, (16,  576, 384), dtype=torch.int64)
# print(part_inx[1, :])
# print(x[0, part_inx[0, :]].size())
#
# print(x[:, 0].unsqueeze(1).size())
# labels = labels[:, 0]
# print(labels)
# B = 4
# a = torch.stack([labels == labels[i] for i in range(B)]).float()
# # a = torch.stack([labels == labels[i] for i in range(B)])
# # a = [labels == labels[i] for i in range(B)]
# print(a)

# a = torch.ones(4)
# # print(a.size())
# print(a)

# a = 0.0005
# print(a)
# print(a/10)
# print('test')
# a = ''
# if a is None:
#     print("1")
# elif a == '':
#     print('2')

# a = 0.0000000034234
# print(format(a,'.1e'))
# state_dict = torch.load('/home/ubuntu/xu/cct_14_7x2_384_imagenet.pth')
# print(state_dict['classifier.fc.weight'].size())
# print(state_dict['classifier.fc.bias'].size())
# b = state_dict['classifier.fc.bias']
# # b = rearrange(b, 'x y -> y x')

# labels_a = torch.Tensor(16, 3, 384, 384)
# print(labels_a.size())
# a = torch.nn.Linear(576, 1)
# c = a(labels_a)
# print(c.size())

# c = torch.matmul(F.softmax(a(labels_a), dim=2).transpose(-1, -2), labels_a).squeeze(-2)
# print(c.size())
# # c = rearrange(c, 'x y -> y x')
# print(c.size())
# num_classes = 200
# ori_fc_weight = state_dict['classifier.fc.weight']
# ori_fc_bias = state_dict['classifier.fc.bias']
# a, b = ori_fc_weight.size()
#
# ori_fc_weight = rearrange(ori_fc_weight, 'x y -> y x')
# fc = torch.nn.Linear(a, num_classes)
# fc_weight = fc(ori_fc_weight)
# fc_weight = rearrange(fc_weight, 'x y -> y x')
# print(fc_weight.size())
#
# b = ori_fc_bias.size(0)
# fc = torch.nn.Linear(b, num_classes)
# fc_bias = fc(ori_fc_bias)
# state_dict['classifier.fc.bias'] = fc_bias
# print(fc_bias.size())
# print(state_dict['classifier.positional_emb'].size())
# print(type(state_dict))
# parameter = torch.nn.Parameter(torch.zeros(1, 1, 384),
#                                requires_grad=True)
# x = state_dict['classifier.positional_emb']
# x = torch.cat((parameter, x), dim=1)
# state_dict['classifier.positional_emb'] = x
# print(x)

# for key in state_dict.keys():
#     print(key)
# a = [1, 2]
# print(isinstance( a,list))

# a = [1, 2, 3]
# b = [3, 4, 5]
#
# for p, i, j in zip(a, b):
#     print(str(i) + '' + str(j))
# localtime = time.time()
# print(type(localtime))
# 1638970500.6771438

# def getTime(a,b):
#     if a>=b:
#         temp = time.localtime(a-b)
#     else:
#         temp = time.localtime(b-a)
#     print(time.asctime(temp))
#
# getTime(1639459592.9832702,1639482812.74201)
# getTime(1639102424.2114468,1639102537.7518182)
# getTime(1639113418.9794886,1639114406.913501)
# getTime(1639060568.7568767,1639061559.9770453)

# 一轮多47秒

# if True and False:
#     print(1)
# else:
#     print(2)

# timestr1 = time.asctime(time.localtime(1638971269.13979))
# timestr2 = time.asctime(time.localtime(1638971382.3969326))
# print(timestr2-timestr1)
# Wed Dec  8 13:33:07 2021
# Wed Dec  8 13:35:00 2021

# 1.83

# Tue Dec  7 07:44:16 2021
# Tue Dec  7 07:46:08 2021

# def lr_lambda(step, warmup_steps=500, t_total=10000, cycles=.5, last_epoch=-1):
#     if step < warmup_steps:
#         return float(step) / float(max(1.0, warmup_steps))
#     # progress after warmup
#     progress = float(step - warmup_steps) / float(max(1, t_total - warmup_steps))
#     return max(1e-5, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))
#
#
# for i in range(1000):
#     a = lr_lambda(i)
#     print(a)
# model_list = timm.list_models()
# # print(len(model_list), model_list[:3])
# # Results 541 ['adv_inception_v3', 'botnet26t_256', 'botnet50ts_256']
# # print(True)
# for name in model_list:
#     if 'vit' in name:
#         print(name)
# a = 0.00020247304066628831
# a = format(a, '.1e')
# b = 3e-5
# print(type(b))
# print(type(a))

# labels_a = torch.randn(16, 3, 384, 384)
# print(labels_a.size())
# #
# a = nn.Conv2d(3, 768,
#               kernel_size=16,
#               stride=16,
#               )  # torch.Size([16, 64, 192, 192])
# b = nn.Identity()
# c = nn.MaxPool2d(kernel_size=3,
#                  stride=2,
#                  padding=1)
# o = a(labels_a)
# print(o.size())
#
# input = torch.randn(20, 16, 50, 100)
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# output = m(input)
# print(output.size())  #  torch.Size([20, 33, 28, 100])
