import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary


# 层归一化层，根据 ViT 的结构，transformer 块中，在多头 attention 层和 MLP 层之前都要先做一个层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 注意，是层归一化，而不是批归一化
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 前馈层，一个两层的 MLP，应该就是 transformer 块中的 MLP 层
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 多头 Attention 层
class Attention(nn.Module):
    def __init__(self, in_dim, heads=8, d_model=64, dropout=0.):
        """

        :param dim: 小片嵌入的维数
        :param heads: 多头 attention 的头数
        :param dim_head: 这个应该就是 d_model，transformer 中 Q、K、V 的维数
        :param dropout:
        """
        super().__init__()
        inner_dim = d_model * heads
        project_out = not (heads == 1 and d_model == in_dim)  # 这里如果只有一个头的话就不用对每个 attention 的结果做连接了

        self.heads = heads  # 多头 attention 的头数
        self.scale = d_model ** -0.5  # QK^T 的量化因子——根号 d

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias=False)  # 这是用来把 x 变成 q k v 的操作，高级写法，原来的三层这里一层就解决了

        # 多个 attention 的结果最后还是用线性变换的方式连接在一起的
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """

        :param x: 小片嵌入：[B, N, dim]
        :return:
        """
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # qkv：[B, N, inner_dim*3]，然后应该是在最后一维上分别拆分出了 q k v，形状均为 [B, N, inner_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 得到 q k v：[B, h, n, dim_head]

        # QK^T/d^0.5，这个矩阵乘法他直接用爱因斯坦操作完成了
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # q k 点积量化得到 attention 向量：[B, h, N, N]
        attn = self.attend(dots)

        # 同样，用爱因斯坦操作来实现矩阵乘法
        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # attention 向量和 v 相乘，得到 attention 层的输出
        out = rearrange(out, 'b h n d -> b n (h d)')  # 对输出做了个变形：[B, h, N, dim_head] => [B, N, inner_dim]
        out = self.to_out(out)  # [B, N, dim]，形状同输入，所以说 transformer 中 attention 层输出的特征向量和输入向量的维数相同！
        return out


# transformer 层
class Transformer(nn.Module):
    def __init__(self, in_dim, depth, heads, d_model, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # ViT 的构建单元就是一层 attention，一层前馈
            self.layers.append(nn.ModuleList([
                PreNorm(in_dim, Attention(in_dim, heads=heads, d_model=d_model, dropout=dropout)),
                PreNorm(in_dim, FeedForward(in_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            # print(x.shape)
            x = attn(x) + x
            x = ff(x) + x
        return x


class SETR(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, emb_dim, depth, heads,
                 mlp_dim, d_model=64, channels=3, dropout=0., emb_dropout=0.):
        """
        emb_dim: 小片嵌入的维数
        depth: transformer 块的个数
        heads: 多头 attention 的头数
        mlp_dim: transformer 中 mlp 的输出维数，也即由 transformer 提取到的特征向量的维度
        d_model: transformer 中 q、k、v 的维数
        """
        super().__init__()
        self.size = image_size
        h, w = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        ph, pw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        # 要求每个小片的大小必须相同，并且刚好可以把图片完全切分
        assert h % ph == 0 and w % pw == 0, 'Image dimensions must be divisible by the patch size.'

        rn, cn = h // ph, w // pw  # 小片的行数和列数
        num_patches = rn * cn
        patch_dim = channels * ph * pw

        # 嵌入层
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (rn p1) (cn p2) -> b (rn cn) (p1 p2 c)', p1=ph, p2=pw),
            nn.Linear(patch_dim, emb_dim),
        )  # 小片嵌入
        # SETR 中不需要 class token，只有位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))  # 位置嵌入
        self.dropout = nn.Dropout(emb_dropout)

        # transformer 层，编码器
        self.transformer = Transformer(emb_dim, depth, heads, d_model, mlp_dim, dropout)  # 输出形状为 [B, N, mlp_dim]

        # 输出层，解码器
        self.decoder = nn.Sequential(
            # 先对特征向量做变形，[B, N, D] => [B, D, H/ps, W/ps]，ps 即 patch_size
            Rearrange('b (rn cn) d -> b d rn cn', rn=rn, cn=cn),  # [B, N, 30, 30]
            nn.Conv2d(mlp_dim, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),  # 到这里，输出形状为 [B, 256, 30, 30]
            # 需做 16 倍上采样
            nn.Upsample(size=self.size, mode="bilinear", align_corners=False)
        )

    def forward(self, img):
        # 嵌入层，对输入图片进行嵌入
        x = self.to_patch_embedding(img)  # [B, C, H, W] => [B, N, dim]
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer 层，depth 个 transformer 块
        x = self.transformer(x)

        # 输出层，这里将 N+1 个 dmodel 维的小片特征向量做平均，作为整张图片最终的特征图，然后再送入 MLP 中进行最后的分类
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SETR(
        image_size=480,
        patch_size=16,
        num_classes=10,
        emb_dim=128,
        depth=6,
        heads=8,
        mlp_dim=128
    ).to(device)
    summary(model, (3, 480, 480))  # 打印模型每一层的输出
    x = torch.randn([1, 3, 480, 480]).to(device)
    print(model(x).shape)
