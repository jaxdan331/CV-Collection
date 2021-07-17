import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential()
        # 卷积层一
        # [batch_size, 1, 28, 28] => [batch_size, 6, 28, 28]
        self.model.add_module('conv1', nn.Conv2d(1, 6, 3, padding=1))
        # [batch_size, 6, 28, 28] => [batch_size, 6, 14, 14]
        self.model.add_module('pool1', nn.MaxPool2d(2, 2))
        # 卷积层二
        # [batch_size, 6, 14, 14] => [batch_size, 16, 10, 10]
        self.model.add_module('conv2', nn.Conv2d(6, 16, 5))
        # [batch_size, 16, 10, 10] => [batch_size, 16, 5, 5]
        self.model.add_module('pool2', nn.MaxPool2d(2, 2))
        # 全连接层
        # [batch_size, 400] => [batch_size, 120]
        self.model.add_module('cf1', nn.Linear(16 * 5 * 5, 120))
        # [batch_size, 120] => [batch_size, 84]
        self.model.add_module('cf2', nn.Linear(120, 84))
        # [batch_size, 84] => [batch_size, 10]
        self.model.add_module('cf3', nn.Linear(84, 10))
        self.model.add_module('softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = x
        for name, module in self.model.named_children():
            # print('module: ', name)
            if name == 'cf1':
                output = output.view(-1, 16 * 5 * 5)
            # print(output.size())
            output = module(output)
        return output


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential()
        # MNIST 是单通道的，而原版的 AlexNet 输入是三通道的，因此要做出修改，每层卷积的通道数都要变成原来的 1/3
        # v = w - k + 2p + 1，要注意的是，若卷积核大小为 3*3 (k=3)，则当 padding=1 时，输入输出图片的大小相同
        # 卷积层一
        self.model.add_module('conv1', nn.Conv2d(1, 64, 5, padding=2))  # [batch_size, 64, 28, 28]
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(2, 2))  # [batch_size, 64, 14, 14]
        # 卷积层二
        self.model.add_module('conv2', nn.Conv2d(64, 192, 3, padding=1))  # [batch_size, 192, 14, 14]
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(2, 2))  # [batch_size, 192, 7, 7]
        # 卷积层三
        self.model.add_module('conv3', nn.Conv2d(192, 384, 3, padding=1))  # [batch_size, 384, 7, 7]
        self.model.add_module('relu3', nn.ReLU())
        # 卷积层四
        self.model.add_module('conv4', nn.Conv2d(384, 256, 3, padding=1))  # [batch_size, 256, 7, 7]
        self.model.add_module('relu4', nn.ReLU())
        # 卷积层五
        self.model.add_module('conv5', nn.Conv2d(256, 256, 3, padding=1))  # [batch_size, 256, 7, 7]
        self.model.add_module('relu5', nn.ReLU())
        self.model.add_module('pool3', nn.MaxPool2d(2, 2))  # [batch_size, 256, 3, 3]
        # 全连接层
        self.model.add_module('fc1', nn.Linear(256 * 3 * 3, 1024))
        self.model.add_module('fc2', nn.Linear(1024, 512))
        self.model.add_module('fc3', nn.Linear(512, 10))
        self.model.add_module('softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = x
        for name, module in self.model.named_children():
            if name == 'fc1':
                output = output.view(-1, 256 * 3 * 3)
            # print(module)
            # print(output.size())
            output = module(output)
        return output


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = nn.Sequential()
        # MNIST 是单通道的，而原版的 AlexNet 输入是三通道的，因此要做出修改，每层卷积的通道数都要变成原来的 1/3
        # v = w - k + 2p + 1，要注意的是，若卷积核大小为 3*3 (k=3)，则当 padding=1 时，输入输出图片的大小相同
        # 卷积层一
        self.model.add_module('conv1', nn.Conv2d(1, 64, 3, padding=1))  # [batch_size, 32, 28, 28]
        self.model.add_module('relu1', nn.ReLU())
        # 卷积层二
        self.model.add_module('conv2', nn.Conv2d(64, 64, 3, padding=1))  # [batch_size, 64, 28, 28]
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(2, 2))  # [batch_size, 64, 14, 14]
        # 卷积层三
        self.model.add_module('conv3', nn.Conv2d(64, 128, 3, padding=1))  # [batch_size, 128, 14, 14]
        self.model.add_module('relu3', nn.ReLU())
        # 卷积层四
        self.model.add_module('conv4', nn.Conv2d(128, 128, 3, padding=1))  # [batch_size, 128, 14, 14]
        self.model.add_module('relu4', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(2, 2))  # [batch_size, 128, 7, 7]
        # 卷积层五
        self.model.add_module('conv5', nn.Conv2d(128, 256, 3, padding=1))  # [batch_size, 256, 7, 7]
        self.model.add_module('relu5', nn.ReLU())
        # 卷积层六
        self.model.add_module('conv6', nn.Conv2d(256, 256, 3, padding=1))  # [batch_size, 256, 7, 7]
        self.model.add_module('relu6', nn.ReLU())
        self.model.add_module('pool3', nn.MaxPool2d(2, 2))  # [batch_size, 256, 3, 3]
        # 全连接层
        self.model.add_module('fc1', nn.Linear(256 * 3 * 3, 1024))
        self.model.add_module('fc2', nn.Linear(1024, 512))
        self.model.add_module('fc3', nn.Linear(512, 10))
        self.model.add_module('softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = x
        for name, module in self.model.named_children():
            if name == 'fc1':
                output = output.view(-1, 256 * 3 * 3)
            # print(module)
            # print(output.size())
            output = module(output)
        return output


# ResNet 的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model(x)
        shape = x.size()
        # 如果 x 的通道数和该层卷积块的目标通道数不同则需要修改
        if shape[1] == 1:
            x = x.expand(shape[0], self.out_channels, shape[2], shape[3])
        elif shape[1] != self.out_channels:
            x = torch.cat([x, x], dim=1)
            # x = x.expand(shape[0], self.out_channels, shape[2], shape[3])
        return self.relu(x1 + x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('block1', ResidualBlock(1, 64))  # [batch_size, 64, 28, 28]
        self.model.add_module('block2', ResidualBlock(64, 128))  # [batch_size, 128, 28, 28]
        self.model.add_module('block3', ResidualBlock(128, 256))  # [batch_size, 256, 28, 28]
        self.model.add_module('pool', nn.AvgPool2d(2, 2))  # [batch_size, 256, 14, 14]
        self.model.add_module('fc', nn.Linear(256 * 14 * 14, 10))
        self.model.add_module('softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        output = x
        for name, module in self.model.named_children():
            if name == 'fc':
                output = output.view(-1, 256 * 14 * 14)
            # print(name)
            # print(output.size())
            output = module(output)
            # print(output.size())
        return output


class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # 这里要注意一下，F.relu 和 nn.ReLU 的区别是，前者是一个数学函数，后者一个层
        # 因此前者可以像一个函数一样直接使用，而后者只能 relu = nn.ReLU，然后 y = relu(x)
        return F.relu(x)


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        # 1x1 卷积
        self.conv1x1 = Conv2dBN(in_channels, 64, kernel_size=1)
        # 3x3 卷积
        self.conv3x3 = Conv2dBN(64, 96, kernel_size=1)
        # 5x5 卷积
        self.conv5x5 = Conv2dBN(64, 96, kernel_size=1)
        # 最大池化，池化核大小为 3x3, stride、padding 均为 1，保证输出与输入大小相同
        self.pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        x1 = self.conv1x1(x)  # [batch_size, 64, 28, 28]
        x2 = self.conv3x3(self.conv1x1(x))  # [batch_size, 96, 28, 28]
        x3 = self.conv5x5(self.conv1x1(x))  # [batch_size, 96, 28, 28]
        x4 = self.conv1x1(self.pool(x))  # [batch_size, 64, 28, 28]
        return torch.cat([x1, x2, x3, x4], 1)  # [batch_size, 64 * 2 + 96 * 2 = 320, 28, 28]


class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.model = nn.Sequential()
        # 卷积层一
        self.model.add_module('conv1', nn.Conv2d(1, 16, 3, padding=1))  # [batch_size, 16, 28, 28]
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(2, 2))  # [batch_size, 16, 14, 14]
        # 卷积层二
        self.model.add_module('conv2', nn.Conv2d(16, 32, 3, padding=1))  # [batch_size, 32, 14, 14]
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(2, 2))  # [batch_size, 32, 7, 7]
        # Inception
        self.model.add_module('inception', Inception(32))  # [batch_size, 320, 7, 7]
        # 全连接
        self.model.add_module('fc1', nn.Linear(320 * 7 * 7, 512))
        self.model.add_module('fc2', nn.Linear(512, 10))
        self.model.add_module('softmax', nn.LogSoftmax(1))

    def forward(self, x):
        output = x
        for name, module in self.model.named_children():
            if name == 'fc1':
                output = output.view(-1, 320 * 7 * 7)
            # print(name)
            # print(output.size())
            output = module(output)
            # print(output.size())
        return output


# helpers
def pair(t):
    # t 要么是一个 tuple: [int, int]，要么是一个整数 int，代表小片的尺寸（方形）
    # 这里应该学习一下人家的写法，估计 pytorch 中 int/tuple 型的参数都是这样写的
    return t if isinstance(t, tuple) else (t, t)


# 层归一化层，根据 ViT 的结构，transformer 块中，在多头 attention 层和 MLP 层之前都要先做一个层归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """

        :param dim: 小片嵌入的维数
        :param heads: 多头 attention 的头数
        :param dim_head: 这个应该就是 d_model，transformer 中 Q、K、V 的维数
        :param dropout:
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)  # 这里如果只有一个头的话就不用对每个 attention 的结果做连接了

        self.heads = heads  # 多头 attention 的头数
        self.scale = dim_head ** -0.5  # QK^T 的量化因子——根号 d

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 这是用来把 x 变成 q k v 的操作，高级写法，原来的三层这里一层就解决了

        # 多个 attention 的结果最后还是用线性变换的方式连接在一起的
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
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
        return self.to_out(out)  # 输出：[B, N, dim]，形状同输入，又可以做下一个 attention 层的输入了


# transformer 层
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # ViT 的构建单元就是一层 attention，一层前馈
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        # 观察这个构造函数的接口给的，又有 image_size 又有 patch_size，就好像模型内部会自动将输入图片划分为小片一样
        super().__init__()
        h, w = pair(image_size)
        ph, pw = pair(patch_size)

        # 要求每个小片的大小必须相同，并且刚好可以把图片完全切分
        assert h % ph == 0 and w % pw == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (h // ph) * (w // pw)
        patch_dim = channels * ph * pw
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 获取输入图片的嵌入向量
        self.to_patch_embedding = nn.Sequential(
            # 下面的 hn = h // ph，wn = w // pw，分别是小片的行数和列数，(hn p1)=h，(wn p2)=w，(hn wn)=num_patches
            # 这个操作就是将 [B, C, W, H] 的输入图片张量转换成 [B, N, P^2C] 的张量，然后再对后者做线性变换，转换成 [B, N, D] 的张量
            Rearrange('b c (hn p1) (wn p2) -> b (hn wn) (p1 p2 c)', p1=ph, p2=pw),
            nn.Linear(patch_dim, dim),
        )  # 小片嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 类别嵌入，就是一个 D 维的可学习的向量（参数）
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置嵌入

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()  # 单位矩阵，根本就没有对输入进行改变，要这一层干嘛？

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # 嵌入层，对输入图片进行嵌入
        x = self.to_patch_embedding(img)  # [B, C, H, W] => [B, N, dim]
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, dim]
        # print(x.shape, self.pos_embedding.shape)
        # print(self.pos_embedding[:, :(n + 1)].shape)
        # x += self.pos_embedding[:, :(n + 1)]
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer 层，depth 个 transformer 块
        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViT(
        image_size=480,
        patch_size=16,
        num_classes=10,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=128
    ).to(device)
    summary(model, (3, 480, 480))  # 打印模型每一层的输出
