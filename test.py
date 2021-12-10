import torch
import time
import math
import timm

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
# labels = torch.randint(5, (4, 10), dtype=torch.int64)
# print(labels)
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

def getTime(a,b):
    if a>=b:
        temp = time.localtime(a-b)
    else:
        temp = time.localtime(b-a)
    print(time.asctime(temp))

getTime(1638971269.13979,1638971382.3969326)
getTime(1639056565.1461518,1639056725.6944604)

#一轮多47秒

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
