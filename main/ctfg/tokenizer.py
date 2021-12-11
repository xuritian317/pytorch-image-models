import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False,
                 is_conv=True, img_size=448, embedding_dim=768):
        super(Tokenizer, self).__init__()
        self.is_conv = is_conv
        if is_conv:
            n_filter_list = [n_input_channels] + \
                            [in_planes for _ in range(n_conv_layers - 1)] + \
                            [n_output_channels]
            self.conv_layers = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                              kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride),
                              padding=(padding, padding), bias=conv_bias),
                    nn.Identity() if activation is None else activation(),
                    nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                 stride=pooling_stride,
                                 padding=pooling_padding) if max_pool else nn.Identity()
                )
                    for i in range(n_conv_layers)
                ])
        else:
            img_size = _pair(img_size)
            patch_size = _pair(16)
            slide_step = 12
            self.n_patches = ((img_size[0] - patch_size[0]) // slide_step + 1) * (
                    (img_size[1] - patch_size[1]) // slide_step + 1)
            self.conv_layers = nn.Conv2d(in_channels=n_input_channels,
                                         out_channels=embedding_dim,
                                         kernel_size=patch_size,
                                         stride=(slide_step, slide_step))

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        # a = self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
        # b = self.n_patches
        # print("\n\n************\n\n")
        # print(a)
        # print(b)
        if self.is_conv:
            return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]
        else:
            return self.n_patches

    def forward(self, x):
        conv_output = self.conv_layers(x)
        output = self.flattener(conv_output).transpose(-2, -1)
        return output

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TextTokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 embedding_dim=300,
                 n_output_channels=128,
                 activation=None,
                 max_pool=True,
                 *args, **kwargs):
        super(TextTokenizer, self).__init__()

        self.max_pool = max_pool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_output_channels,
                      kernel_size=(kernel_size, embedding_dim),
                      stride=(stride, 1),
                      padding=(padding, 0), bias=False),
            nn.Identity() if activation is None else activation(),
            nn.MaxPool2d(
                kernel_size=(pooling_kernel_size, 1),
                stride=(pooling_stride, 1),
                padding=(pooling_padding, 0)
            ) if max_pool else nn.Identity()
        )

        self.apply(self.init_weight)

    def seq_len(self, seq_len=32, embed_dim=300):
        return self.forward(torch.zeros((1, seq_len, embed_dim)))[0].shape[1]

    def forward_mask(self, mask):
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones(
            (1, 1, self.conv_layers[0].kernel_size[0]),
            device=mask.device,
            dtype=torch.float)
        new_mask = F.conv1d(
            new_mask, cnn_weight, None,
            self.conv_layers[0].stride[0], self.conv_layers[0].padding[0], 1, 1)
        if self.max_pool:
            new_mask = F.max_pool1d(
                new_mask, self.conv_layers[2].kernel_size[0],
                self.conv_layers[2].stride[0], self.conv_layers[2].padding[0], 1, False, False)
        new_mask = new_mask.squeeze(1)
        new_mask = (new_mask > 0)
        return new_mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)
        x = x if mask is None else x * self.forward_mask(mask).unsqueeze(-1).float()
        return x, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
