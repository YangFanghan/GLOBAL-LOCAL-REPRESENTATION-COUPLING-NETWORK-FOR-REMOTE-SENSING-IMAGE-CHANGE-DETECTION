import torch
import torch.nn as nn
import math


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h

class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 1, stride = 1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, input1,input2):
        batch_size, channels, height, width = input1.shape#输入两张图一样的尺寸

        #构建两组qkv
        # input: B, C, H, W -> q: B, H * W, C // 8
        #input: B, C, H, W -> k: B, C // 8, H * W
        #input: B, C, H, W -> v: B, C, H * W
        q1 = self.query(input1).view(batch_size, -1, height * width).permute(0, 2, 1)
        k1 = self.key(input1).view(batch_size, -1, height * width)
        v1 = self.value(input1).view(batch_size, -1, height * width)
        
        q2 = self.query(input2).view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key(input2).view(batch_size, -1, height * width)
        v2 = self.value(input2).view(batch_size, -1, height * width)
        
        #att1
        #q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix_1 = torch.bmm(q2, k1)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix_1 = self.softmax(attn_matrix_1)#经过一个softmax进行缩放权重大小.
        out_1 = torch.bmm(v1, attn_matrix_1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out_1 = out_1.view(*input1.shape)
        out_1 = self.gamma * out_1.contiguous() + input1.contiguous()
        
        #att2
        attn_matrix_2 = torch.bmm(q1, k2)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix_2 = self.softmax(attn_matrix_2)#经过一个softmax进行缩放权重大小.
        out_2 = torch.bmm(v2, attn_matrix_1.permute(0, 2, 1))  #tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out_2 = out_2.view(*input2.shape)
        out_2 = self.gamma * out_2.contiguous() + input2.contiguous()
        
        #att_t1
        return out_1 , out_2
   

          

class ECAlayer(nn.Module):
    def __init__(self, channel, gamma=2,bias=1):
        super(ECAlayer, self).__init__()
        # x: input features with shape [b, c, h, w]
        self.channel=channel
        self.gamma=gamma
        self.bias=bias
 
        k_size=int(abs((math.log(self.channel , 2)+self.bias)/self.gamma))#(log底数2,c + 1) / 2 ---->(log 2,512 + 1)/2 = 5
        k_size= k_size if k_size%2 else k_size+1  #按照输入通道数自适应的计算卷积核大小
 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.avg_pool(x) # 基于全局空间信息的特征描述符
        # b,c,1,1
        # 变换维度，使张量可以进入卷积层
        y = self.conv(y.squeeze(-1).transpose(-1, -2))#压缩一个维度后，在转换维度
        #b,c,1,1 ----》 b,c,1 ----》 b,1,c      可以理解为输入卷积的batch，只有一个通道所以维度是1，c理解为序列卷积的特征个数
        y = y.transpose(-1, -2).unsqueeze(-1)
        #b,1,c ----》 b,c,1 ----》 b,c,1,1
        # 多尺度信息融合
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class CA(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CA, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return x

class SpatialAttention(nn.Module):           # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)*x
        return out

class GAM_Attention(nn.Module):  
    def __init__(self, in_channels, out_channels, rate=4):  
        super(GAM_Attention, self).__init__()  

        self.channel_attention = nn.Sequential(  
            nn.Linear(in_channels, int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )  
      
        self.spatial_attention = nn.Sequential(  
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),  
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(out_channels)  
        )  
      
    def forward(self, x):  
        b, c, h, w = x.shape  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)  
      
        x = x * x_channel_att  
      
        x_spatial_att = self.spatial_attention(x).sigmoid()  
        out = x * x_spatial_att  
      
        return out  


class DESpatialAttention(nn.Module):           # Spatial Attention Module
    def __init__(self):
        super(DESpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1,x2):
        x = torch.abs(x1 - x2)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out1 = out * x1 +x1
        out2 = out * x2 +x2
        return out1, out2


class MSAttention(nn.Module):           # 多尺度注意力模块
    def __init__(self,ifspatialattention=False):
        super(MSAttention, self).__init__()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.maxpool_8 = nn.MaxPool2d(kernel_size=3, stride=8, padding=1)
        self.maxpool_16 = nn.MaxPool2d(kernel_size=3, stride=16, padding=1)
        self.softmax = nn.Softmax(dim = -1)
        self.sat = SpatialAttention()
        self.ifsat = ifspatialattention
        self.ifat = ifspatialattention
        #self.downsample = F.interpolate(x, scale_factor=0.5)
        #dilation空洞卷积
    def forward(self, x):
        feature_x = x.view(x.shape[0], -1, x.shape[2] * x.shape[3])#16,32,4096
        feature_x2 = self.maxpool_2(x)
        feature_x4 = self.maxpool_4(x)
        feature_x8 = self.maxpool_8(x)
        feature_x16 = self.maxpool_16(x)
        if self.ifsat:
            feature_x2 = self.sat(feature_x2)
            feature_x4 = self.sat(feature_x4)
            feature_x8 = self.sat(feature_x8)
            feature_x16 = self.sat(feature_x16)
        
        feature_x2 = feature_x2.view(feature_x2.shape[0], -1, feature_x2.shape[2] * feature_x2.shape[3])
        feature_x4 = feature_x4.view(feature_x4.shape[0], -1, feature_x4.shape[2] * feature_x4.shape[3])
        feature_x8 = feature_x8.view(feature_x8.shape[0], -1, feature_x8.shape[2] * feature_x8.shape[3])     
        feature_x16 = feature_x16.view(feature_x16.shape[0], -1, feature_x16.shape[2] * feature_x16.shape[3])
        
        feature = torch.cat([feature_x2,feature_x4,feature_x8,feature_x16],dim=2)#16 32 1360
        feature_t = feature.permute(0,2,1)#16 1360 32
        feature_att = self.softmax(torch.bmm(feature,feature_t))
        out = (torch.bmm(feature_att,feature_x)).view(*x.shape) + x
        return out
        
class MSChangeAttention(nn.Module):           # 多尺度注意力模块
    def __init__(self,ifchangeatt=False):
        super(MSChangeAttention, self).__init__()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)
        self.maxpool_8 = nn.MaxPool2d(kernel_size=3, stride=8, padding=1)
        self.maxpool_16 = nn.MaxPool2d(kernel_size=3, stride=16, padding=1)
        self.softmax = nn.Softmax(dim = -1)
        self.att = selfattention(32)
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        self.conv_fusion = nn.Conv2d(32*4, 32, kernel_size = 3 , stride = 1,padding=2,dilation=2)#全局感受野
        
        #self.downsample = F.interpolate(x, scale_factor=0.5)
        #dilation空洞卷积
    def forward(self, x1,x2):
        feature_x1_x2 = self.maxpool_2(x1)
        feature_x1_x4 = self.maxpool_4(x1)
        feature_x1_x8 = self.maxpool_8(x1)
        feature_x1_x16 = self.maxpool_16(x1)
        
        feature_x2_x2 = self.maxpool_2(x2)
        feature_x2_x4 = self.maxpool_4(x2)
        feature_x2_x8 = self.maxpool_8(x2)
        feature_x2_x16 = self.maxpool_16(x2)
        #多尺度changeatt
        out_x1_x2 , out_x2_x2 = self.att(feature_x1_x2,feature_x2_x2)
        out_x1_x4 , out_x2_x4 = self.att(feature_x1_x4,feature_x2_x4)
        out_x1_x4 , out_x2_x4 = self.upsamplex2(out_x1_x4), self.upsamplex2(out_x2_x4)
        out_x1_x8 , out_x2_x8 = self.att(feature_x1_x8,feature_x2_x8)
        out_x1_x8 , out_x2_x8 = self.upsamplex4(out_x1_x8) ,self.upsamplex4(out_x2_x8)
        out_x1_x16 , out_x2_x16 = self.att(feature_x1_x16,feature_x2_x16)
        out_x1_x16 , out_x2_x16 = self.upsamplex2(self.upsamplex4(out_x1_x16)) , self.upsamplex2(self.upsamplex4(out_x2_x16))
        
        out1 = self.conv_fusion(torch.cat([out_x1_x2 , out_x1_x4 , out_x1_x8 , out_x1_x16],dim=1))
        out2 = self.conv_fusion(torch.cat([out_x2_x2 , out_x2_x4 , out_x2_x8 , out_x2_x16],dim=1))
        out1 = self.upsamplex2(out1)
        out2 = self.upsamplex2(out2)

        return out1,out2
      
class MSChangeAttention_plus(nn.Module):           # 多尺度注意力模块
    def __init__(self,ifchangeatt=False):
        super(MSChangeAttention_plus, self).__init__()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax = nn.Softmax(dim = -1)
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.conv_fusion = nn.Conv2d(32*2, 32, kernel_size = 3 , stride = 1,padding=2,dilation=2)#全局感受野
        self.att = selfattention(32)
        #self.downsample = F.interpolate(x, scale_factor=0.5)
        #dilation空洞卷积
    def forward(self, x1,x2):
        feature_x1_x2 = self.maxpool_2(x1)
        feature_x2_x2 = self.maxpool_2(x2)

        #多尺度changeatt
        out_x1_x2 , out_x2_x2 = self.att(feature_x1_x2,feature_x2_x2)
        
        feature_x1_x4 = self.maxpool_2(out_x1_x2)
        feature_x2_x4 = self.maxpool_2(out_x2_x2)
        
        out_x1_x4 , out_x2_x4 = self.att(feature_x1_x4,feature_x2_x4)
        
        feature_x1_x8 = self.maxpool_2(out_x1_x4)
        feature_x2_x8 = self.maxpool_2(out_x2_x4)
        
        out_x1_x8 , out_x2_x8 = self.att(feature_x1_x8,feature_x2_x8)
        
        feature_x1_x16 = self.maxpool_2(out_x1_x8)
        feature_x2_x16 = self.maxpool_2(out_x2_x8)
        
    
        out_x1_x16 , out_x2_x16 = self.att(feature_x1_x16,feature_x2_x16)
        
        out_x1_x16 , out_x2_x16 = self.upsamplex2(out_x1_x16) , self.upsamplex2(out_x2_x16)
        out1_x1_x8 = self.conv_fusion(torch.cat([out_x1_x16 , out_x1_x8],dim=1))
        out1_x2_x8 = self.conv_fusion(torch.cat([out_x2_x16 , out_x2_x8],dim=1))
        
        out_x1_x8 , out_x2_x8 = self.upsamplex2(out_x1_x8) , self.upsamplex2(out_x2_x8)
        out1_x1_x4 = self.conv_fusion(torch.cat([out_x1_x8 , out_x1_x4],dim=1))
        out1_x2_x4 = self.conv_fusion(torch.cat([out_x2_x8 , out_x2_x4],dim=1))
        
        out_x1_x4 , out_x2_x4 = self.upsamplex2(out_x1_x4) , self.upsamplex2(out_x2_x4)
        out1= self.conv_fusion(torch.cat([out_x1_x4 , out_x1_x2],dim=1))
        out2 = self.conv_fusion(torch.cat([out_x2_x4 , out_x2_x2],dim=1))
        
        # out1 = self.conv_fusion(torch.cat([out_x1_x2 , out_x1_x4 , out_x1_x8 , out_x1_x16],dim=1))
        # out2 = self.conv_fusion(torch.cat([out_x2_x2 , out_x2_x4 , out_x2_x8 , out_x2_x16],dim=1))
        out1 = self.upsamplex2(out1)
        out2 = self.upsamplex2(out2)

        return out1,out2
