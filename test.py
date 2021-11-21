import torch

a = [1, 2, 3, 4, 5]
# b = a[:-1]
# c = a[:-2]
# print(b)
# print(c)

for blk in a[:-2]:
    print(blk)

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
