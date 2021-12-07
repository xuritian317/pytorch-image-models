import torch
import time
import math


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
# state_dict = torch.load('/hy-tmp/ctfg_fork/output/train/5e-4_128B_83.6/model_pretrained_224.pth.tar')
# for key in state_dict.keys():
#     print(key)


# a = [1, 2, 3]
# b = [3, 4, 5]
#
# for p, i, j in zip(a, b):
#     print(str(i) + '' + str(j))
# localtime = time.time()
# timestr = time.asctime(time.localtime(1638802320.8617198))
# print(type(localtime))
# print(timestr)


def lr_lambda(step, warmup_steps=500, t_total=10000, cycles=.5, last_epoch=-1):
    if step < warmup_steps:
        return float(step) / float(max(1.0, warmup_steps))
    # progress after warmup
    progress = float(step - warmup_steps) / float(max(1, t_total - warmup_steps))
    return max(1e-5, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))


for i in range(1000):
    a = lr_lambda(i)
    print(a)
