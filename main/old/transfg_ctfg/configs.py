import ml_collections
import torch.nn as nn


# CCT-7/3x1
# CCT-7/7x2
# CCT-14t/7x2

def get_cct731_config():
    """Returns the CCT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'overlap'
    config.slide_step = 12
    # config.embedding_dim = 256
    config.hidden_size = 256
    # config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072

    config.attention_dropout_rate = 0.1
    config.dropout_rate = 0.1
    config.stochastic_depth_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    config.n_conv_layers = 1
    config.in_planes = 64

    config.kernel_size = 3

    config.num_layers = 7

    config.num_heads = 4
    config.mlp_ratio = 2
    config.mlp_dim = config.hidden_size * config.mlp_ratio
    config.stride = max(1, (3 // 2) - 1)
    config.padding = max(1, (3 // 2))

    config.conv_bias = False

    config.activation = nn.ReLU

    config.pooling_kernel_size = 3
    config.pooling_stride = 2
    config.pooling_padding = 1

    config.max_pool = True

    config.seq_pool = True
    return config


def get_cct772_config():
    """Returns the CCT configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'overlap'
    config.slide_step = 12
    config.hidden_size = 256
    # config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072
    config.attention_dropout_rate = 0.1
    config.dropout_rate = 0.1
    config.stochastic_depth_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    config.n_conv_layers = 2
    config.in_planes = 64

    config.kernel_size = 7

    config.num_layers = 7

    config.num_heads = 4
    config.mlp_ratio = 2
    config.mlp_dim = config.hidden_size * config.mlp_ratio
    config.stride = max(1, (7 // 2) - 1)
    config.padding = max(1, (7 // 2))

    config.conv_bias = False

    config.activation = nn.ReLU

    config.pooling_kernel_size = 3
    config.pooling_stride = 2
    config.pooling_padding = 1

    config.max_pool = True
    config.seq_pool = True
    return config


def get_cct1472_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'overlap'
    config.slide_step = 12
    config.hidden_size = 384
    # config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072

    config.attention_dropout_rate = 0.1
    config.dropout_rate = 0.1

    config.classifier = 'token'
    config.representation_size = None

    config.n_conv_layers = 2
    config.in_planes = 64

    config.kernel_size = 7

    config.num_layers = 14

    config.num_heads = 6
    config.mlp_ratio = 3

    config.mlp_dim = config.hidden_size * config.mlp_ratio

    config.stride = max(1, (7 // 2) - 1)
    config.padding = max(1, (7 // 2))

    config.conv_bias = False
    config.activation = nn.ReLU
    config.pooling_kernel_size = 3
    config.pooling_stride = 2
    config.pooling_padding = 1

    config.max_pool = True

    config.seq_pool = True

    config.stochastic_depth_rate = 0.1
    return config


def get_cct1672_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'overlap'
    config.slide_step = 12
    config.hidden_size = 384
    # config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072

    config.attention_dropout_rate = 0.1
    config.dropout_rate = 0.1

    config.classifier = 'token'
    config.representation_size = None

    config.n_conv_layers = 2
    config.in_planes = 64

    config.kernel_size = 7

    config.num_layers = 16

    config.num_heads = 6
    config.mlp_ratio = 3

    config.mlp_dim = config.hidden_size * config.mlp_ratio

    config.stride = max(1, (7 // 2) - 1)
    config.padding = max(1, (7 // 2))

    config.conv_bias = False
    config.activation = nn.ReLU
    config.pooling_kernel_size = 3
    config.pooling_stride = 2
    config.pooling_padding = 1

    config.max_pool = True

    config.seq_pool = True

    config.stochastic_depth_rate = 0.1
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'overlap'
    config.slide_step = 12
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


def get_b16_add_conv_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 12
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    config.in_planes = 64
    config.n_conv_layers = 2
    config.kernel_size = 7
    config.stride = max(1, (7 // 2) - 1)
    config.padding = max(1, (7 // 2))
    config.activation = nn.ReLU
    config.conv_bias = False
    config.pooling_kernel_size = 3
    config.pooling_stride = 2
    config.pooling_padding = 1
    config.max_pool = True
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config
