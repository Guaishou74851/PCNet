import torch, dct
from torch import nn
import torch.nn.functional as F
from utils import *
from scunet import ConvTransBlock

minus = lambda r1, r2: [r1[i] - r2[i] for i in range(len(r1))]

class RB(nn.Module):
    def __init__(self, nf, t):
        super().__init__()
        self.body = ConvTransBlock(nf//2, nf//2, nf//2, 16, 0, t, float('inf'))

    def forward(self, x):
        return self.body(x)

class Stage(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor([1.0]))
        self.s = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1),
            RB(nf, 'W'), RB(nf, 'SW'),
            nn.Conv2d(nf, nf, 3, padding=1),
        )

    def forward(self, x):
        x, y, A, AT, SS = x
        x = F.pixel_shuffle(x, 2)
        z = x[:, :2]
        z1, H = SS.fuse(x, self.s)
        z = z - self.rho * AT(minus(A(z1), y))
        x = H + torch.cat([z, x[:, 2:]], dim=1)
        x = F.pixel_unshuffle(x, 2)
        x = x + self.body(x)
        return x, y, A, AT, SS

class SkipConv(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.body = nn.Conv2d(nf, nf, 3, padding=1)
        self.scale = nn.Sequential(nn.Conv2d(2, nf, 1), nn.ReLU(True), nn.Conv2d(nf, nf, 1))

    def forward(self, x):
        x, cs_ratio = x
        return self.scale(cs_ratio) * self.body(x), cs_ratio

class SS(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.head = nn.Conv2d(1, nf//4, 3, padding=1)
        self.body = nn.Sequential(*[SkipConv(nf//4) for _ in range(5)])
        self.tail = nn.Conv2d(nf//4, 2, 3, padding=1)
    
    def forward(self, x):
        return self.tail(self.body([self.head(x), self.cs_ratio])[0])

    def fuse(self, x, s):
        x1 = x[:, 2:3]
        x2 = s[0] * x + self.head(x1)
        x2 = self.body([x2, self.cs_ratio])[0]
        x3 = self.tail(x2)
        return x3, s[1] * x2

class Net(nn.Module):
    def __init__(self, nb, B, nf):
        super().__init__()
        self.B = B
        self.N = B * B
        self.SS = SS(nf)
        U, S, V = torch.linalg.svd(torch.randn(self.N, self.N))
        self.A_weight_G = nn.Parameter(U.mm(V).reshape(self.N, 1, B, B), requires_grad=False)
        self.head = nn.Conv2d(2, nf, 6, padding=2, stride=2)
        self.body = nn.Sequential(*[Stage(nf) for _ in range(nb)])
        self.tail = nn.Sequential(nn.Conv2d(nf, 4, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x, q_G, q_DCT):
        b, c, h, w = x.shape
        n = h * w
        h_B, w_B = h//self.B, w//self.B
        cs_ratio_G = (q_G / self.N).view(b,1,1,1)
        cs_ratio_DCT = (q_DCT / self.N).view(b,1,1,1)
        cs_ratio = torch.cat([cs_ratio_G, cs_ratio_DCT], dim=1).view(b,2,1,1)
        self.SS.cs_ratio = cs_ratio
        perm = torch.randperm(n, device=x.device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(n, device=x.device)
        A_weight_G = self.A_weight_G[torch.randperm(self.N, device=x.device)].to(x.device)
        mask_G = (torch.arange(self.N,device=x.device).view(1,self.N).expand(b,self.N) < q_G.view(b,1)).view(b,self.N,1,1)
        mask_DCT = (torch.arange(self.N,device=x.device).view(1,self.N).expand(b,self.N) < q_DCT.view(b,1)).view(b,self.N,1,1)
        DCT_x, DCT_y = get_zigzag_truncated_indices(h, w, n)
        A_G = lambda z: F.conv2d(z.reshape(b,c,n)[:,:,perm].reshape(b,c,h,w), A_weight_G, stride=self.B) * mask_G
        A_DCT = lambda z: dct.dct_2d(z)[:, :, DCT_x, DCT_y].reshape(b, self.N, h_B, w_B) * mask_DCT
        AT_G = lambda z: F.conv_transpose2d(z, A_weight_G, stride=self.B).reshape(b,c,n)[:,:,perm_inv].reshape(b,c,h,w)
        def AT_DCT(z):
            z_ = torch.empty(b, 1, h, w, device=x.device)
            z_[:, :, DCT_x, DCT_y] = z.reshape(b, 1, -1)
            return dct.idct_2d(z_)
        A = lambda z: [A_G(z[:,0:1]), A_DCT(z[:,1:2])]
        AT = lambda z: torch.cat([AT_G(z[0]), AT_DCT(z[1])], dim=1)
        y = A(self.SS(x))
        x = self.head(AT(y))
        x = self.body([x, y, A, AT, self.SS])[0]
        return self.tail(x)
    
if __name__ == '__main__':
    device = 'cuda'
    model = Net(20, 32, 32).to(device)
    x = torch.rand(16, 1, 128, 128).to(device)
    cs_ratio, N = 0.1, 1024
    q = (torch.tensor([[0.1 * 1024]], device=device)).ceil().expand(16, 1)
    q_G = torch.rand(16, 1, device=device)
    q_G = (q_G * q).round()
    q_DCT = q - q_G
    x_out = model(x, q_G, q_DCT)
    print(x.shape)
    print(x_out.shape)
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#Param.', param_cnt/1e6, 'M')