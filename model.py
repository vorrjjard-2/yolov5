import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

    def forward(self, x):
        return x

class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out, e=0.5, shortcut=True, g=1):
        super().__init__()
        c_ = int(c_in * e)
        self.cv1 = Conv(c_in, c_, 1, 1, None) #reduce dimensionality first
        self.cv2 = Conv(c_, c_out, 3, 1, None, g=g) #heavy lifting, project back to original dims (or close)
        self.add = shortcut and c_in == c_out 

    def forward(self, x):
        return self.cv2(self.cv1(x)) + x if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c_in, c_out, n=1, e=0.5, shortcut=True):
        super().__init__()
        self.c_ = c_in * e
        
        
    def forward(self, x):
        return x

class SPPF(nn.Module):
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        
    def forward(self, x):
        return x


class CSPDarknetBackbone(nn.Module):
    def __init__(self, c_in=3, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        
    def forward(self, x):        
        return x


