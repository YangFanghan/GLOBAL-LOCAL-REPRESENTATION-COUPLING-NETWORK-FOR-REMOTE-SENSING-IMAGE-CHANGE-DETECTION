from audioop import bias
from mimetypes import init
from turtle import forward
from torch import nn
import torch 
import numpy as np
from torch.nn import init
import math
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm



class External_attention(nn.Module):
    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)

        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x = self.linear_1(attn)

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

if __name__ in '__main__':
    img = torch.randn(4, 6, 256, 256)
    EA = External_attention(6)
    out = EA(img)
    print(out.shape)
























# class ExternalAttention(nn.Module):  #### xia da version
#     def __init__(self, d_model, S):
#         super().__init__()
#         self.mk = nn.Linear(d_model, S, bias=False)
#         self.mv = nn.Linear(S, d_model, bias=False)
#         self.softmax=nn.Softmax(dim=1)
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant(m.weight, 1)
#                 init.constant(m.bias, 0)

#             elif isinstance(m, nn.BatchNorm1d):
#                 init.constant(m.weight, 1)
#                 init.constant(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, queries):
#         attn = self.mk(queries) # bs, n, S
#         attn = self.softmax(attn) # bs, n, S
#         attn = attn/torch.sum(attn, dim=2, keepdim=True) # bs, n, S
#         out = self.mv(attn) # bs, n, d_model
#         return out

# if __name__=='__main__':
#     input=torch.randn(4, 256, 256)
#     ea = ExternalAttention(d_model=512, S=8)
#     output=ea(input)
#     print(output.shape)

