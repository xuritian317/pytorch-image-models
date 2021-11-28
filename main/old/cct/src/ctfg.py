import torch
from torch.hub import load_state_dict_from_url, load
import torch.nn as nn
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
from .utils.helpers import pe_check
import os

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

model_urls = {
    'ctfg_14_7x2_224':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
    'ctfg_14_7x2_384':
        'cct_14_7x2_384_imagenet.pth',
    'ctfg_14_7x2_384_fl':
        'http://ix.cs.uoregon.edu/~alih/compact-transformers/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth',
}


class CTFG(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(CTFG, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x, flag=False):
        x = self.tokenizer(x)
        return self.classifier(x, flag)


def _ctfg(arch, pretrained, progress,
          num_layers, num_heads, mlp_ratio, embedding_dim,
          kernel_size=3, stride=None, padding=None,
          *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CTFG(num_layers=num_layers,
                 num_heads=num_heads,
                 mlp_ratio=mlp_ratio,
                 embedding_dim=embedding_dim,
                 kernel_size=kernel_size,
                 stride=stride,
                 padding=padding,
                 *args, **kwargs)

    if pretrained:
        # pass
        # TODO the pretrain function is to be implementation
        print('ctfg_old pretrained')

        checkpoint_path = kwargs.get('pretrained_dir', os.path.join('/', model_urls[arch]))
        is_changeSize = kwargs.get('is_changeSize', False)

        if not is_changeSize:
            # state_dict = load_state_dict_from_url(model_urls[arch], model_dir=model_dir,
            #                                       progress=True)

            state_dict = torch.load(checkpoint_path)

            state_dict = pe_check(model, state_dict)

            state_dict['classifier.fc.weight'] = model.classifier.fc.weight
            state_dict['classifier.fc.bias'] = model.classifier.fc.bias

            num = str(num_layers - 1)
            state_dict['classifier.part_select.last_block.pre_norm.weight'] = state_dict[
                'classifier.blocks.' + num + '.pre_norm.weight']

            state_dict['classifier.part_select.last_block.pre_norm.bias'] = state_dict[
                'classifier.blocks.' + num + '.pre_norm.bias']

            state_dict['classifier.part_select.last_block.linear1.weight'] = state_dict[
                'classifier.blocks.' + num + '.linear1.weight']

            state_dict['classifier.part_select.last_block.linear1.bias'] = state_dict[
                'classifier.blocks.' + num + '.linear1.bias']

            state_dict['classifier.part_select.last_block.norm1.weight'] = state_dict[
                'classifier.blocks.' + num + '.norm1.weight']

            state_dict['classifier.part_select.last_block.norm1.bias'] = state_dict[
                'classifier.blocks.' + num + '.norm1.bias']

            state_dict['classifier.part_select.last_block.linear2.weight'] = state_dict[
                'classifier.blocks.' + num + '.linear2.weight']

            state_dict['classifier.part_select.last_block.linear2.bias'] = state_dict[
                'classifier.blocks.' + num + '.linear2.bias']

            state_dict['classifier.part_select.last_block.self_attn.proj.weight'] = state_dict[
                'classifier.blocks.' + num + '.self_attn.proj.weight']

            state_dict['classifier.part_select.last_block.self_attn.proj.bias'] = state_dict[
                'classifier.blocks.' + num + '.self_attn.proj.bias']

            state_dict[
                'classifier.part_select.last_block.self_attn.qkv.weight'] = state_dict[
                'classifier.blocks.' + num + '.self_attn.qkv.weight']

            model.load_state_dict(state_dict)
            print("Loaded from checkpoint '{}'".format(checkpoint_path))
        else:
            # raise RuntimeError(f'Variant {arch} does not yet have pretrained weights.')
            state_dict = torch.load(checkpoint_path)['state_dict']
            state_dict = pe_check(model, state_dict)
            model.load_state_dict(state_dict)

    return model


def ctfg_7(arch, pretrained, progress, *args, **kwargs):
    return _ctfg(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                 *args, **kwargs)


def ctfg_14(arch, pretrained, progress, *args, **kwargs):
    return _ctfg(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                 *args, **kwargs)


@register_model
def ctfg_7_3x1_32(pretrained=False, progress=False,
                  img_size=32, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return ctfg_7('ctfg_7_3x1_32', pretrained, progress,
                  kernel_size=3, n_conv_layers=1,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_3x1_32_sine(pretrained=False, progress=False,
                       img_size=32, positional_embedding='sine', num_classes=10,
                       *args, **kwargs):
    return ctfg_7('ctfg_7_3x1_32_sine', pretrained, progress,
                  kernel_size=3, n_conv_layers=1,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_3x1_32_c100(pretrained=False, progress=False,
                       img_size=32, positional_embedding='learnable', num_classes=100,
                       *args, **kwargs):
    return ctfg_7('ctfg_7_3x1_32_c100', pretrained, progress,
                  kernel_size=3, n_conv_layers=1,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_3x1_32_sine_c100(pretrained=False, progress=False,
                            img_size=32, positional_embedding='sine', num_classes=100,
                            *args, **kwargs):
    return ctfg_7('ctfg_7_3x1_32_sine_c100', pretrained, progress,
                  kernel_size=3, n_conv_layers=1,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_3x2_32(pretrained=False, progress=False,
                  img_size=32, positional_embedding='learnable', num_classes=10,
                  *args, **kwargs):
    return ctfg_7('ctfg_7_3x2_32', pretrained, progress,
                  kernel_size=3, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_3x2_32_sine(pretrained=False, progress=False,
                       img_size=32, positional_embedding='sine', num_classes=10,
                       *args, **kwargs):
    return ctfg_7('ctfg_7_3x2_32_sine', pretrained, progress,
                  kernel_size=3, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=102,
                   *args, **kwargs):
    return ctfg_7('ctfg_7_7x2_224', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_7_7x2_224_sine(pretrained=False, progress=False,
                        img_size=224, positional_embedding='sine', num_classes=102,
                        *args, **kwargs):
    return ctfg_7('ctfg_7_7x2_224_sine', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)


@register_model
def ctfg_14_7x2_224(pretrained=False, progress=False,
                    img_size=224, positional_embedding='learnable', num_classes=1000,
                    *args, **kwargs):
    return ctfg_14('ctfg_14_7x2_224', pretrained, progress,
                   kernel_size=7, n_conv_layers=2,
                   img_size=img_size, positional_embedding=positional_embedding,
                   num_classes=num_classes,
                   *args, **kwargs)


@register_model
def ctfg_14_7x2_384(pretrained=False, progress=False,
                    img_size=384, positional_embedding='learnable', num_classes=1000,
                    *args, **kwargs):
    return ctfg_14('ctfg_14_7x2_384', pretrained, progress,
                   kernel_size=7, n_conv_layers=2,
                   img_size=img_size, positional_embedding=positional_embedding,
                   num_classes=num_classes,
                   *args, **kwargs)


@register_model
def ctfg_14_7x2_384_fl(pretrained=False, progress=False,
                       img_size=384, positional_embedding='learnable', num_classes=102,
                       *args, **kwargs):
    return ctfg_14('ctfg_14_7x2_384_fl', pretrained, progress,
                   kernel_size=7, n_conv_layers=2,
                   img_size=img_size, positional_embedding=positional_embedding,
                   num_classes=num_classes,
                   *args, **kwargs)
