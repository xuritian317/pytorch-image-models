import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath


class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

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


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        a = self.pre_norm(src)
        # print("\n\n****a******\n\n")
        # print(a.size()) torch.Size([16, 576, 384])

        x, weights = self.self_attn(a)
        # print("\n\n****x******\n\n")
        # print(x.size()) torch.Size([16, 576, 384])

        # print("\n\n****weights******\n\n")
        # print(weights.size())  torch.Size([16, 6, 576, 576])

        x = self.drop_path(x)
        src = src + x
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src, weights


# PSM
class PartAttention(Module):
    def __init__(self, last_block):
        super(PartAttention, self).__init__()
        self.last_block = last_block

    def forward(self, attn_weights, x):
        # print('attn_weights.size()')  # 13
        # print(attn_weights[0].size())  # torch.Size([16, 6, 576, 576]) torch.Size([16, 6, 577, 577])
        # print('x.size()')
        # print(x.size())  # torch.Size([16, 576, 384]) torch.Size([16, 577, 384])

        length = len(attn_weights)

        last_map = attn_weights[0]

        for i in range(1, length):
            last_map = torch.matmul(attn_weights[i], last_map)

        last_map = last_map[:, :, 0, :]
        # last_map = last_map[:, :, 0, 1:]

        # print('last_map.size()')
        # print(last_map.size())  # torch.Size([16, 6, 576])

        _, part_inx = last_map.max(2)  # 2 dim

        # part_inx = part_inx + 1

        parts = []
        B, _ = part_inx.shape
        for i in range(B):
            parts.append(x[i, part_inx[i, :]])

        parts = torch.stack(parts).squeeze(1)

        concat = torch.cat((x[:, 0].unsqueeze(1), parts), dim=1)

        # 最后一层
        part_states, _ = self.last_block(concat)
        return part_states


class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 is_psm=True):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.is_psm = is_psm

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)

        if self.is_psm:
            self.part_select = PartAttention(self.blocks[-1])

        self.apply(self.init_weight)

    def forward(self, x, flag=False):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        if self.is_psm:
            attn_weights = []
            for blk in self.blocks[:-1]:
                x, weights = blk(x)
                attn_weights.append(weights)

            part_states = self.part_select(attn_weights, x)
            # print("\n\n**********\n\n")
            # print(part_states.size()) torch.Size([16, 7, 384])
            x = self.norm(part_states)

        else:
            for blk in self.blocks:
                x, _ = blk(x)
            x = self.norm(x)

        if self.seq_pool:
            # print(x.size()) torch.Size([16, 6, 384])
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        part_token = x

        # print(x.size())  16*384
        # print(self.fc.weight.size())#    384*200
        x = self.fc(x)

        if flag:
            return x, part_token
        else:
            return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
