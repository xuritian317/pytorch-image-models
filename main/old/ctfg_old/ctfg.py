# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Dropout, Linear, LayerNorm
from einops import rearrange, repeat
from .configs import *

from timm.models.registry import register_model
from timm.data import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.vision_transformer import _cfg
from timm.models.layers.helpers import to_2tuple

logger = logging.getLogger(__name__)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x, weights = self.self_attn(self.pre_norm(src))
        x = self.drop_path(x)
        src = src + x
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src, weights


# PSM
class PartAttention(nn.Module):
    def __init__(self):
        super(PartAttention, self).__init__()

    def forward(self, x):
        length = len(x)

        last_map = x[0]

        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)

        last_map = last_map[:, :, 0, :]

        _, max_inx = last_map.max(2)

        return _, max_inx


# TransFG的 Encoder
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.layer_num = config.num_layers

        dpr = [x.item() for x in torch.linspace(0, config.stochastic_depth_rate, config.num_layers)]

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(config=config, d_model=config.hidden_size, nhead=config.num_heads,
                                    dim_feedforward=config.mlp_dim, dropout=config.dropout_rate,
                                    attention_dropout=config.attention_dropout_rate, drop_path_rate=dpr[i])
            for i in range(config.num_layers - 1)])

        self.part_select = PartAttention()
        self.part_norm = LayerNorm(config.hidden_size)

        self.part_layer = TransformerEncoderLayer(config=config, d_model=config.hidden_size, nhead=config.num_heads,
                                                  dim_feedforward=config.mlp_dim, dropout=config.dropout_rate,
                                                  attention_dropout=config.attention_dropout_rate,
                                                  drop_path_rate=dpr[config.num_layers - 1])

    def forward(self, hidden_states):
        attn_weights = []
        # 通过 Encoder

        for layer in self.blocks:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)

        _, part_inx = self.part_select(attn_weights)

        # part_inx = part_inx + 1
        parts = []
        B, num = part_inx.shape
        for i in range(B):
            parts.append(hidden_states[i, part_inx[i, :]])

        parts = torch.stack(parts).squeeze(1)
        concat = torch.cat((hidden_states[:, 0].unsqueeze(1), parts), dim=1)

        # 最后一层
        part_states, _ = self.part_layer(concat)
        part_encoded = self.part_norm(part_states)

        return part_encoded


# 分割图片与可学习的位置参数
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, input_size):
        super(Embeddings, self).__init__()
        self.hybrid = None

        img_size = to_2tuple(input_size[1])
        self.n_conv_layers = config.n_conv_layers

        n_filter_list = [input_size[0]] + \
                        [config.in_planes for _ in range(config.n_conv_layers - 1)] + \
                        [config.hidden_size]

        self.patch_embeddings = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(config.kernel_size, config.kernel_size),
                          stride=(config.stride, config.stride),
                          padding=(config.padding, config.padding),
                          bias=config.conv_bias),

                # activation is relu
                nn.Identity() if config.activation is None else config.activation(),

                nn.MaxPool2d(kernel_size=config.pooling_kernel_size,
                             stride=config.pooling_stride,
                             padding=config.pooling_padding) if config.max_pool else nn.Identity()
            )
                for i in range(config.n_conv_layers)
            ])

        self.seq_pool = config.seq_pool

        n_patches = self.sequence_length(3, img_size[0], img_size[0])
        if self.seq_pool:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size), requires_grad=True)
            nn.init.trunc_normal_(self.position_embeddings, std=0.2)
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size),
                                                    requires_grad=True)
        self.dropout = Dropout(0)

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        a = torch.zeros(1, n_channels, height, width)
        a = self.patch_embeddings(a)

        a = rearrange(a, 'b c h w -> b c (h w)')
        # a = rearrange(a, 'b h w -> b w h')

        return a.shape[2]

    def forward(self, img):
        x = self.patch_embeddings(img)

        x = rearrange(x, 'b c h w -> b c (h w)')

        x = rearrange(x, 'b h w -> b w h')

        if not self.seq_pool:
            B = img.shape[0]
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
            x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Transformer(nn.Module):
    def __init__(self, config, input_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, input_size=input_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded


class CTFG(nn.Module):
    def __init__(self, config, input_size=(3, 384, 384), num_classes=21843, smoothing_value=0):
        super(CTFG, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value

        # transformer
        self.transformer = Transformer(config, input_size)

        self.part_head = Linear(config.hidden_size, num_classes)

        self.attention_pool = nn.Linear(config.hidden_size, 1)
        self.seq_pool = config.seq_pool
        self.apply(self.init_weight)

    def forward(self, x, labels=None):
        part_tokens = self.transformer(x)
        # 16 7 384

        if self.seq_pool:
            part_tokens = torch.matmul(F.softmax(self.attention_pool(part_tokens), dim=1).transpose(-1, -2),
                                       part_tokens).squeeze(-2)
        else:
            part_tokens = part_tokens[:, 0]

        part_logits = self.part_head(part_tokens)

        # 混合损失计算
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)

            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))

            contrast_loss = con_loss(part_tokens, labels.view(-1))

            loss = part_loss + contrast_loss
            return loss, part_logits

        else:
            return part_logits

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from(self, weights,num_classes,em):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings[0][0].weight.copy_(weights["tokenizer.conv_layers.0.0.weight"])
            self.transformer.embeddings.patch_embeddings[1][0].weight.copy_(weights["tokenizer.conv_layers.1.0.weight"])
            self.transformer.embeddings.position_embeddings.copy_(weights["classifier.positional_emb"])
            self.attention_pool.weight.copy_(weights["classifier.attention_pool.weight"])
            self.attention_pool.bias.copy_(weights["classifier.attention_pool.bias"])

            self.transformer.encoder.part_norm.weight.copy_(weights["classifier.norm.weight"])
            self.transformer.encoder.part_norm.bias.copy_(weights["classifier.norm.bias"])



            encoder_layers = self.transformer.encoder.layer_num - 1
            encoder_layers = 14 if encoder_layers > 14 else encoder_layers
            for i in range(encoder_layers):
                self.transformer.encoder.blocks[i].pre_norm.weight.copy_(
                    weights["classifier.blocks." + str(i) + ".pre_norm.weight"])
                self.transformer.encoder.blocks[i].pre_norm.bias.copy_(
                    weights["classifier.blocks." + str(i) + ".pre_norm.bias"])
                self.transformer.encoder.blocks[i].norm1.weight.copy_(
                    weights["classifier.blocks." + str(i) + ".norm1.weight"])
                self.transformer.encoder.blocks[i].norm1.bias.copy_(
                    weights["classifier.blocks." + str(i) + ".norm1.bias"])
                self.transformer.encoder.blocks[i].linear1.weight.copy_(
                    weights["classifier.blocks." + str(i) + ".linear1.weight"])
                self.transformer.encoder.blocks[i].linear1.bias.copy_(
                    weights["classifier.blocks." + str(i) + ".linear1.bias"])
                self.transformer.encoder.blocks[i].linear2.weight.copy_(
                    weights["classifier.blocks." + str(i) + ".linear2.weight"])
                self.transformer.encoder.blocks[i].linear2.bias.copy_(
                    weights["classifier.blocks." + str(i) + ".linear2.bias"])
                self.transformer.encoder.blocks[i].self_attn.proj.weight.copy_(
                    weights["classifier.blocks." + str(i) + ".self_attn.proj.weight"])
                self.transformer.encoder.blocks[i].self_attn.proj.bias.copy_(
                    weights["classifier.blocks." + str(i) + ".self_attn.proj.bias"])
                self.transformer.encoder.blocks[i].self_attn.qkv.weight.copy_(
                    weights["classifier.blocks." + str(i) + ".self_attn.qkv.weight"])


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    # neg_cos_matrix[neg_cos_matrix < 0] = 0
    neg_cos_matrix = neg_cos_matrix.clamp(min=0.0)
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 200, 'input_size': (3, 384, 384), 'smoothing_value': 0,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        **kwargs
    }


__all__ = [
    'ctfg_1472'
]


def _create_ctfg(variant, pretrained=False, default_cfg=None, **kwargs):
    pass


@register_model
def ctfg_1472(pretrained=False, **kwargs):
    config = get_cct1472_config()
    default_cfg = _cfg('', **kwargs)

    input_size = kwargs.get('input_size', default_cfg['input_size'])
    num_classes = kwargs.get('num_classes', default_cfg['num_classes'])
    # smoothing_value = kwargs.get('smoothing_value', default_cfg.smoothing_value)

    model = CTFG(config, input_size=input_size, num_classes=num_classes, smoothing_value=0)

    model.default_cfg = default_cfg

    return model
